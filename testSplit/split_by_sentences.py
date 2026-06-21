# -*- coding: utf-8 -*-
r"""
split_by_sentences.py
=====================

Forced-align a Bulgarian audiobook chapter to its transcript and cut the audio
into one WAV clip per sentence, producing a ``text,audioFile`` CSV manifest.

Pipeline
--------
1. **Normalise** ``testText1.txt`` into synchronized representations:
     * word-only text for aeneas alignment;
     * punctuation-preserving text for training metadata.
   The word-only result is written to ``testText1.normalized.txt`` because that
   is the plain-text format aeneas expects. Punctuation is never discarded from
   the CSV training transcript.
2. **Align** the sentence list to ``01_Pista01.mp3`` with **aeneas** (forced
   alignment) -> begin/end timestamp for every sentence.  Saved as ``syncmap.json``.
   (Use ``--reuse-syncmap`` to skip this slow step and re-cut from a saved map.)
3. **Refine + cut.** aeneas's espeak-ng reference voice is robotic, so its exact
   boundaries land ~1-2s off (mid-word). We therefore decode the audio to mono
   WAV, detect the silences (pauses between sentences) and **snap each boundary
   to the nearest silence**, which makes the cuts clean. Fragments aeneas could
   not place in real audio (zero-duration -- i.e. the transcript is longer than
   the audio) are dropped. Clips are written to ``clips/0001.wav`` ...
4. **Write** ``metadata.csv`` with columns ``text,audioFile`` mapping each
   kept sentence to its clip.

Requirements
------------
* ``aeneas``           (the forced aligner)
* ``espeak``/``espeak-ng``  (aeneas uses it to synthesise the reference text; the
                         Bulgarian voice ``bg`` must be present)
* ``ffmpeg``           (audio decoding; must be on PATH or passed via --ffmpeg)
* ``numpy``            (aeneas dependency)

Install (Linux / Google Colab -- easiest):
    sudo apt-get install -y espeak espeak-data libespeak-dev ffmpeg
    pip install numpy aeneas

Install (Windows -- how the ML conda env on this machine was set up):
    aeneas's C extensions don't build cleanly here, but they're only a speed
    optimisation -- aeneas falls back to pure Python. Install it without them:

        conda install -n ML -c conda-forge ffmpeg
        python -m pip install lxml beautifulsoup4
        # build with all C extensions disabled (no compiler needed):
        set AENEAS_WITH_CDTW=False & set AENEAS_WITH_CMFCC=False & set AENEAS_WITH_CEW=False
        python -m pip install --no-build-isolation aeneas

    espeak-ng (not on conda-forge) was unpacked without admin via an MSI
    administrative extract into the env:

        msiexec /a espeak-ng.msi /qn TARGETDIR=<env>\espeak

    so that <env>\espeak\espeak-ng.exe and <env>\espeak\espeak-ng-data\ exist.
    This script auto-detects ffmpeg at {prefix}\Library\bin and espeak-ng at
    {prefix}\espeak, and sets ESPEAK_DATA_PATH itself, when run with that env's
    Python. Override with --ffmpeg / --espeak / --espeak-data if your layout
    differs.

Note: with the pure-Python aligner this takes a few minutes on a ~30 min file.

Usage
-----
    python split_by_sentences.py                 # uses the defaults below
    python split_by_sentences.py --sample-rate 0 # keep the source sample rate
    python split_by_sentences.py --espeak "D:/Anaconda/envs/ML/espeak/espeak-ng.exe"
"""

import argparse
import csv
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Make Cyrillic output safe on a cp1252 Windows console.
# ---------------------------------------------------------------------------
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover - older Pythons / redirected streams
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


# ---------------------------------------------------------------------------
# Reuse the same dependency-free dual normalizer as MFA and synthesis.
# ---------------------------------------------------------------------------
def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_bg = _load_module(
    "bulgarian_dual_normalization", REPO_ROOT / "bulgarian_normalization.py"
)
normalize_for_mfa = _bg.normalize_for_mfa
normalize_with_punctuation = _bg.normalize_with_punctuation


# ---------------------------------------------------------------------------
# Step 1 -- text normalisation
# ---------------------------------------------------------------------------
# A sentence ends at . ! ? or … (possibly several in a row), followed by space.
# The source file is not hard-wrapped: each paragraph is one physical line, so a
# sentence never spans two lines and we can segment line by line. Lines without a
# terminator (headings, the canon lines, "Пролог") become a single fragment.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


def normalise_to_sentence_pairs(raw_text: str) -> list[tuple[str, str]]:
    """Return ``(word_only_alignment_text, punctuated_training_text)`` pairs."""
    sentences: list[tuple[str, str]] = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        for chunk in _SENTENCE_SPLIT_RE.split(line):
            word_only = normalize_for_mfa(chunk)
            punctuated = normalize_with_punctuation(chunk)
            if word_only:
                if normalize_for_mfa(punctuated) != word_only:
                    raise RuntimeError("Dual normalization produced different words")
                sentences.append((word_only, punctuated))
    return sentences


def normalise_to_sentences(raw_text: str) -> list[str]:
    """Backward-compatible word-only view used by older callers/tests."""
    return [word_only for word_only, _ in normalise_to_sentence_pairs(raw_text)]


# ---------------------------------------------------------------------------
# Step 2 -- forced alignment with aeneas
# ---------------------------------------------------------------------------
def align(audio_path: Path, text_path: Path, syncmap_path: Path,
          language: str, tts: str, tts_path: str | None) -> list[tuple[float, float]]:
    """Run aeneas forced alignment and return [(begin, end), ...] in seconds,
    one entry per input line, in order. Also writes the sync map to JSON.

    `tts` selects the synthesiser aeneas uses for the reference audio
    ("espeak-ng", "espeak", "festival"); `tts_path` is the binary's full path.
    """
    from aeneas.executetask import ExecuteTask
    from aeneas.task import Task
    from aeneas.runtimeconfiguration import RuntimeConfiguration

    config_string = "|".join([
        f"task_language={language}",
        "is_text_type=plain",          # one fragment per line
        "os_task_file_format=json",
    ])
    task = Task(config_string=config_string)
    task.audio_file_path_absolute = str(audio_path)
    task.text_file_path_absolute = str(text_path)
    task.sync_map_file_path_absolute = str(syncmap_path)

    # The C extensions (cdtw/cmfcc) aren't built on this install; aeneas falls
    # back to its pure-Python implementations automatically (slower but works).
    rconf = RuntimeConfiguration()
    if tts:
        rconf[RuntimeConfiguration.TTS] = tts
    if tts_path:
        rconf[RuntimeConfiguration.TTS_PATH] = tts_path

    ExecuteTask(task, rconf=rconf).execute()
    task.output_sync_map_file()

    spans: list[tuple[float, float]] = []
    for fragment in task.sync_map_leaves():
        # Skip HEAD/TAIL/no-speech fragments aeneas may insert.
        if not getattr(fragment, "is_regular", True):
            continue
        text = (fragment.text or "").strip()
        if not text:
            continue
        spans.append((float(fragment.begin), float(fragment.end)))
    return spans


# ---------------------------------------------------------------------------
# Step 3 -- snap boundaries to silence and cut the audio
#
# aeneas gets the *gross* (sentence-level) alignment right, but because the
# espeak-ng reference voice sounds nothing like the human narrator, its exact
# boundaries are noisy (they land mid-word ~1-2s off). We therefore keep
# aeneas's boundaries only as anchors and snap each one to the nearest real
# silence (the pause between sentences), which yields clean cuts.
# ---------------------------------------------------------------------------
def decode_to_wav(audio_path: Path, wav_path: Path, ffmpeg: str,
                  sample_rate: int) -> None:
    """Decode any input to a single 16-bit mono WAV."""
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(audio_path), "-ac", "1"]
    if sample_rate and sample_rate > 0:
        cmd += ["-ar", str(sample_rate)]
    cmd += ["-sample_fmt", "s16", str(wav_path)]
    subprocess.run(cmd, check=True)


def load_wav_mono(wav_path: Path) -> tuple[np.ndarray, int]:
    """Load a 16-bit mono WAV as float32 in [-1, 1]."""
    with wave.open(str(wav_path), "rb") as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm, sr


def silence_centers(pcm: np.ndarray, sr: int, thr: float,
                    min_sil: float) -> tuple[np.ndarray, float]:
    """Return (centres of silence intervals, last_speech_time). A silence is a
    run of >= ``min_sil`` seconds whose frame RMS stays below ``thr``."""
    win, hop = int(0.020 * sr), int(0.010 * sr)
    n_frames = 1 + (len(pcm) - win) // hop
    # Fast frame RMS via a prefix sum of squares.
    csum = np.concatenate(([0.0], np.cumsum(pcm.astype(np.float64) ** 2)))
    starts = np.arange(n_frames) * hop
    rms = np.sqrt((csum[starts + win] - csum[starts]) / win)
    times = (starts + win / 2) / sr

    quiet = rms < thr
    centres: list[float] = []
    i = 0
    while i < n_frames:
        if quiet[i]:
            j = i
            while j < n_frames and quiet[j]:
                j += 1
            a, b = times[i], times[j - 1]
            if b - a >= min_sil:
                centres.append(0.5 * (a + b))
            i = j
        else:
            i += 1
    speech_idx = np.where(~quiet)[0]
    last_speech = float(times[speech_idx[-1]]) if speech_idx.size else len(pcm) / sr
    return np.asarray(centres), last_speech


def snap_boundaries(spans: list[tuple[float, float]], centres: np.ndarray,
                    window: float) -> list[tuple[float, float]]:
    """Snap each fragment's end to the nearest silence centre within ``window``
    seconds; clips stay contiguous (a fragment starts where the previous ended).
    Returns the snapped, non-overlapping spans."""
    def nearest(t: float):
        if centres.size == 0:
            return None
        k = int(np.argmin(np.abs(centres - t)))
        return float(centres[k]) if abs(centres[k] - t) <= window else None

    out: list[tuple[float, float]] = []
    first_start = nearest(spans[0][0])
    prev_end = first_start if (first_start is not None and first_start < spans[0][1]) \
        else spans[0][0]
    for i, (begin, end) in enumerate(spans):
        start = prev_end
        snapped = nearest(end)
        if snapped is None or snapped <= start + 0.20:
            end = max(end, start + 0.05)
        else:
            end = snapped
        out.append((start, end))
        prev_end = end
    return out


def write_clips(pcm: np.ndarray, sr: int, spans: list[tuple[float, float]],
                clips_dir: Path) -> list[str]:
    """Write one 16-bit mono WAV per span; return the filenames in order."""
    clips_dir.mkdir(parents=True, exist_ok=True)
    for stale in clips_dir.glob("*.wav"):   # clear clips from a previous run
        stale.unlink()
    names: list[str] = []
    total = len(pcm)
    width = max(4, len(str(len(spans))))
    for index, (begin, end) in enumerate(spans, start=1):
        a = max(0, min(int(round(begin * sr)), total))
        b = max(a, min(int(round(end * sr)), total))
        seg = np.clip(pcm[a:b] * 32768.0, -32768, 32767).astype(np.int16)
        name = f"{index:0{width}d}.wav"
        with wave.open(str(clips_dir / name), "wb") as dst:
            dst.setnchannels(1)
            dst.setsampwidth(2)
            dst.setframerate(sr)
            dst.writeframes(seg.tobytes())
        names.append(name)
    return names


def load_syncmap_spans(syncmap_path: Path) -> list[tuple[float, float]]:
    """Read (begin, end) spans from an existing aeneas JSON sync map, in order."""
    data = json.load(open(syncmap_path, encoding="utf-8"))
    return [(float(f["begin"]), float(f["end"])) for f in data["fragments"]]


# ---------------------------------------------------------------------------
# Locate the external tools and prepare the runtime environment
# ---------------------------------------------------------------------------
def locate_tools(args) -> tuple[str, str | None, str | None]:
    """Resolve ffmpeg / espeak-ng paths and put their dirs (plus
    ESPEAK_DATA_PATH) on the environment so aeneas's subprocess calls find them.

    Auto-detection assumes the script is run with the conda env that has the
    tools installed (``sys.prefix``), matching this machine's ML env layout:
        ffmpeg     -> {prefix}/Library/bin/ffmpeg.exe
        espeak-ng  -> {prefix}/espeak/espeak-ng.exe  (+ espeak-ng-data alongside)
    Any of these can be overridden with --ffmpeg / --espeak / --espeak-data.
    """
    prefix = Path(sys.prefix)

    ffmpeg = args.ffmpeg
    if ffmpeg == "ffmpeg":
        cand = prefix / "Library" / "bin" / "ffmpeg.exe"
        if cand.exists():
            ffmpeg = str(cand)

    espeak = args.espeak
    if espeak is None:
        cand = prefix / "espeak" / "espeak-ng.exe"
        espeak = str(cand) if cand.exists() else None

    espeak_data = args.espeak_data
    if espeak_data is None and espeak:
        cand = Path(espeak).parent
        if (cand / "espeak-ng-data").exists():
            espeak_data = str(cand)

    # Make the tools discoverable to aeneas (it shells out to them).
    for directory in {Path(ffmpeg).parent, Path(espeak).parent if espeak else None}:
        if directory and directory.exists():
            os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
    if espeak_data:
        os.environ["ESPEAK_DATA_PATH"] = espeak_data

    return ffmpeg, espeak, espeak_data


def check_dependencies(ffmpeg: str, espeak: str | None) -> None:
    problems = []
    try:
        import aeneas  # noqa: F401
    except Exception as exc:
        problems.append(f"  - aeneas is not importable ({exc}). See install notes "
                        f"in this file's docstring.")
    if shutil.which(ffmpeg) is None and not Path(ffmpeg).exists():
        problems.append(f"  - ffmpeg not found ('{ffmpeg}'). Install it or pass "
                        f"--ffmpeg <path>.")
    if (espeak is None or not Path(espeak).exists()) \
            and shutil.which("espeak-ng") is None and shutil.which("espeak") is None:
        problems.append("  - espeak-ng not found. Pass --espeak <path to espeak-ng.exe> "
                        "(its espeak-ng-data folder must sit beside it).")
    if problems:
        print("Missing dependencies:", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalise Bulgarian text, force-align it to audio with "
                    "aeneas, and split the audio into per-sentence clips + CSV.")
    parser.add_argument("--text", type=Path, default=SCRIPT_DIR / "testText1.txt",
                        help="Input transcript (default: testText1.txt).")
    parser.add_argument("--audio", type=Path, default=SCRIPT_DIR / "01_Pista01.mp3",
                        help="Input audio (default: 01_Pista01.mp3).")
    parser.add_argument("--outdir", type=Path, default=SCRIPT_DIR,
                        help="Where to write outputs (default: this folder).")
    parser.add_argument("--clips-dir", default="clips",
                        help="Sub-folder for the audio clips (default: clips).")
    parser.add_argument("--csv", default="metadata.csv",
                        help="Output CSV filename (default: metadata.csv).")
    parser.add_argument("--normalized", default="testText1.normalized.txt",
                        help="Normalised one-sentence-per-line text filename.")
    parser.add_argument("--syncmap", default="syncmap.json",
                        help="aeneas sync-map JSON filename.")
    parser.add_argument("--language", default="bul",
                        help="aeneas/espeak language code (default: bul).")
    parser.add_argument("--sample-rate", type=int, default=22050,
                        help="Output sample rate; 0 keeps the source rate "
                             "(default: 22050, the FastSpeech2 BG target).")
    parser.add_argument("--ffmpeg", default="ffmpeg",
                        help="ffmpeg executable (path or name; auto-detected in "
                             "the conda env if left as 'ffmpeg').")
    parser.add_argument("--tts", default="espeak-ng",
                        help="aeneas TTS engine: espeak-ng (default), espeak, festival.")
    parser.add_argument("--espeak", default=None,
                        help="Path to espeak-ng.exe (auto-detected in the conda env).")
    parser.add_argument("--espeak-data", default=None,
                        help="Path to the folder containing espeak-ng-data "
                             "(auto-detected next to --espeak).")
    parser.add_argument("--keep-temp", action="store_true",
                        help="Keep the intermediate full decoded WAV.")
    parser.add_argument("--reuse-syncmap", action="store_true",
                        help="Skip alignment and reuse an existing sync map "
                             "(fast re-cut while tuning the snapping).")
    # Boundary snapping / silence detection
    parser.add_argument("--no-snap", action="store_true",
                        help="Do not snap boundaries to silence (use raw aeneas "
                             "boundaries -- usually cuts mid-word).")
    parser.add_argument("--silence-frac", type=float, default=0.33,
                        help="Silence RMS threshold as a fraction of the audio's "
                             "global RMS (default: 0.33).")
    parser.add_argument("--min-silence", type=float, default=0.12,
                        help="Minimum silence run length in seconds (default: 0.12).")
    parser.add_argument("--snap-window", type=float, default=2.5,
                        help="Max distance (s) to snap a boundary to a silence "
                             "(default: 2.5).")
    parser.add_argument("--min-duration", type=float, default=0.30,
                        help="Drop fragments shorter than this (s); these are the "
                             "ones aeneas could not align to any audio (default: 0.30).")
    args = parser.parse_args()

    if not args.text.exists():
        sys.exit(f"Transcript not found: {args.text}")
    if not args.audio.exists():
        sys.exit(f"Audio not found: {args.audio}")

    ffmpeg, espeak, espeak_data = locate_tools(args)
    check_dependencies(ffmpeg, espeak)
    print(f"      ffmpeg={ffmpeg}")
    print(f"      tts={args.tts} espeak={espeak} data={espeak_data}")

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    normalized_path = outdir / args.normalized
    syncmap_path = outdir / args.syncmap
    clips_dir = outdir / args.clips_dir
    csv_path = outdir / args.csv

    # --- Step 1: normalise -------------------------------------------------
    raw = args.text.read_text(encoding="utf-8")
    sentence_pairs = normalise_to_sentence_pairs(raw)
    if not sentence_pairs:
        sys.exit("No sentences produced from the transcript.")
    sentences = [word_only for word_only, _ in sentence_pairs]
    punctuated_sentences = [punctuated for _, punctuated in sentence_pairs]
    normalized_path.write_text("\n".join(sentences) + "\n", encoding="utf-8")
    print(f"[1/3] Normalised {len(sentences)} sentences -> {normalized_path.name}")

    # --- Step 2: align (or reuse an existing sync map) ---------------------
    if args.reuse_syncmap and syncmap_path.exists():
        print(f"[2/3] Reusing existing sync map -> {syncmap_path.name}")
        spans = load_syncmap_spans(syncmap_path)
    else:
        print(f"[2/3] Forced-aligning with aeneas (language={args.language}) ... "
              f"(pure-Python DTW; this can take a few minutes)")
        spans = align(args.audio, normalized_path, syncmap_path, args.language,
                      args.tts, espeak)
        print(f"      sync map -> {syncmap_path.name}")
    if len(spans) != len(sentences):
        print(f"  ! aeneas returned {len(spans)} fragments for {len(sentences)} "
              f"sentences; pairing the first {min(len(spans), len(sentences))}.",
              file=sys.stderr)
    n = min(len(spans), len(sentences))
    spans = spans[:n]
    sentences = sentences[:n]
    punctuated_sentences = punctuated_sentences[:n]

    # --- Step 3: decode, detect silence, drop orphans, snap, cut ----------
    print("[3/3] Cutting audio into clips ...")
    full_wav = outdir / "_full.wav"
    decode_to_wav(args.audio, full_wav, ffmpeg, args.sample_rate)
    try:
        pcm, sr = load_wav_mono(full_wav)
    finally:
        if not args.keep_temp:
            full_wav.unlink(missing_ok=True)

    global_rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
    thr = args.silence_frac * global_rms
    centres, last_speech = silence_centers(pcm, sr, thr, args.min_silence)
    print(f"      audio {len(pcm)/sr:.0f}s, {len(centres)} silences, "
          f"speech ends at {last_speech:.0f}s")

    # Drop fragments aeneas could not place in real audio (zero-duration tail,
    # or starting past the last speech): these have no corresponding audio.
    kept = [(s, sp) for s, sp in zip(punctuated_sentences, spans)
            if (sp[1] - sp[0]) >= args.min_duration and sp[0] < last_speech - 0.2]
    dropped = len(sentences) - len(kept)
    if not kept:
        sys.exit("No fragments survived filtering -- alignment failed.")
    kept_sentences = [s for s, _ in kept]
    kept_spans = [sp for _, sp in kept]

    if args.no_snap:
        final_spans = kept_spans
    else:
        final_spans = snap_boundaries(kept_spans, centres, args.snap_window)

    names = write_clips(pcm, sr, final_spans, clips_dir)

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["text", "audioFile"])
        for sentence, name in zip(kept_sentences, names):
            writer.writerow([sentence, f"{args.clips_dir}/{name}"])

    if dropped:
        print(f"  ! dropped {dropped}/{len(sentences)} sentences with no matching "
              f"audio (the audio is shorter than the transcript).", file=sys.stderr)
    print(f"      {len(names)} clips -> {clips_dir}{os.sep}")
    print(f"      manifest -> {csv_path.name}")
    print("Done.")


if __name__ == "__main__":
    main()
