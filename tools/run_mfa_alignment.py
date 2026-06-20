#!/usr/bin/env python3
"""Content-addressed, resumable MFA alignment with G2P-backed OOV handling."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def sha256_json(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def artifact_spec(value):
    path = Path(value)
    if path.is_file():
        return {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return {"model": value}


def corpus_description(corpus):
    speakers = []
    utterances = []
    for speaker_dir in sorted(path for path in corpus.iterdir() if path.is_dir()):
        ids = sorted(path.stem for path in speaker_dir.glob("*.wav"))
        labs = {path.stem: path for path in speaker_dir.glob("*.lab")}
        missing = sorted(set(ids) - set(labs))
        if missing:
            raise RuntimeError("Missing labs in {}: {}".format(speaker_dir, missing[:20]))
        speakers.append((speaker_dir, ids))
        for uid in ids:
            lab = labs[uid]
            utterances.append(
                {
                    "id": uid,
                    "speaker": speaker_dir.name,
                    "wav_size": (speaker_dir / "{}.wav".format(uid)).stat().st_size,
                    "lab_sha256": hashlib.sha256(lab.read_bytes()).hexdigest(),
                }
            )
    if not utterances:
        raise RuntimeError("No wav/lab pairs found under {}".format(corpus))
    ids = [item["id"] for item in utterances]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Utterance basenames must be globally unique across speakers")
    return speakers, utterances


def make_batches(speakers, maximum):
    batches = []
    current = []
    count = 0
    for speaker_dir, ids in speakers:
        if current and count + len(ids) > maximum:
            batches.append(current)
            current = []
            count = 0
        current.append((speaker_dir, ids))
        count += len(ids)
    if current:
        batches.append(current)
    return batches


def link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def run_stream(command):
    print("$", " ".join(map(str, command)), flush=True)
    started = time.time()
    process = subprocess.Popen(
        [str(item) for item in command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    print("completed in {:.1f} min".format((time.time() - started) / 60.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="mfa_corpus")
    parser.add_argument(
        "--output", default="preprocessed_data/Bulgarian/TextGrid/Bulgarian"
    )
    parser.add_argument("--work", default="mfa_phone_work")
    parser.add_argument(
        "--dictionary", default="lexicon/bulgarian_mfa_runtime.dict"
    )
    parser.add_argument("--acoustic", default="bulgarian_mfa")
    parser.add_argument("--g2p", default="bulgarian_mfa")
    parser.add_argument("--mfa-command", default="mfa")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--max-utterances", type=int, default=3000)
    parser.add_argument(
        "--max-failed-ratio",
        type=float,
        default=0.01,
        help="Maximum explicit MFA export failures allowed across the corpus",
    )
    parser.add_argument(
        "--reset-output",
        action="store_true",
        help="Delete an old/incompatible TextGrid run before starting",
    )
    args = parser.parse_args()

    corpus = Path(args.corpus)
    output = Path(args.output)
    work = Path(args.work)
    metadata_path = output.parent / "alignment_run.json"

    speakers, utterances = corpus_description(corpus)
    try:
        mfa_version = subprocess.check_output(
            [args.mfa_command, "version"], text=True
        ).strip()
    except Exception as exc:
        raise RuntimeError("Cannot execute MFA command: {}".format(args.mfa_command)) from exc

    run_spec = {
        "corpus_sha256": sha256_json(utterances),
        "utterance_count": len(utterances),
        "dictionary": artifact_spec(args.dictionary),
        "acoustic": artifact_spec(args.acoustic),
        "g2p": artifact_spec(args.g2p),
        "mfa_version": mfa_version,
    }
    run_id = sha256_json(run_spec)[:16]

    if args.reset_output:
        shutil.rmtree(output, ignore_errors=True)
        metadata_path.unlink(missing_ok=True)
    if output.exists() and any(output.glob("*.TextGrid")):
        if not metadata_path.exists():
            raise RuntimeError(
                "Existing TextGrids have no alignment_run.json. Use --reset-output "
                "to prevent mixing the old grapheme/OOV run with this phone run."
            )
        old = json.loads(metadata_path.read_text(encoding="utf-8"))
        if old.get("run_id") != run_id:
            raise RuntimeError(
                "Existing alignment belongs to run {} but current run is {}. "
                "Use --reset-output.".format(old.get("run_id"), run_id)
            )

    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    failed_ids = set()
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous.get("run_id") == run_id:
            failed_ids.update(previous.get("failed_ids", []))
    metadata_path.write_text(
        json.dumps(
            {
                **run_spec,
                "run_id": run_id,
                "status": "in_progress",
                "failed_ids": sorted(failed_ids),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    done_dir = work / "done" / run_id
    done_dir.mkdir(parents=True, exist_ok=True)

    batches = make_batches(speakers, args.max_utterances)
    for index, batch in enumerate(batches, 1):
        expected = sorted(uid for _, ids in batch for uid in ids)
        chunk_spec = {
            "run_id": run_id,
            "expected_ids": expected,
            "max_utterances": args.max_utterances,
        }
        chunk_id = sha256_json(chunk_spec)[:16]
        marker = done_dir / "{}.json".format(chunk_id)
        existing = {path.stem for path in output.glob("*.TextGrid")}
        expected_success = set(expected) - failed_ids
        if expected_success.issubset(existing):
            print("[skip] {}/{} {} ({} TextGrids + {} recorded failures)".format(
                index,
                len(batches),
                chunk_id,
                len(expected_success),
                len(set(expected) & failed_ids),
            ))
            continue

        chunk_root = work / "chunks" / chunk_id
        chunk_corpus = chunk_root / "corpus"
        chunk_output = chunk_root / "output"
        shutil.rmtree(chunk_root, ignore_errors=True)
        chunk_corpus.mkdir(parents=True)

        for speaker_dir, ids in batch:
            destination = chunk_corpus / speaker_dir.name
            destination.mkdir()
            for uid in ids:
                link_or_copy(speaker_dir / "{}.wav".format(uid), destination / "{}.wav".format(uid))
                link_or_copy(speaker_dir / "{}.lab".format(uid), destination / "{}.lab".format(uid))

        command = [
            args.mfa_command,
            "align",
            str(chunk_corpus),
            args.dictionary,
            args.acoustic,
            str(chunk_output),
            "--g2p_model_path",
            args.g2p,
            "--clean",
            "-j",
            str(args.jobs),
        ]
        print("[align] {}/{} {} utterances={}".format(
            index, len(batches), chunk_id, len(expected)
        ))
        run_stream(command)

        produced_paths = list(chunk_output.rglob("*.TextGrid"))
        produced = {path.stem for path in produced_paths}
        missing = sorted(set(expected) - produced)
        extra = sorted(produced - set(expected))
        projected_failures = failed_ids | set(missing)
        failure_ratio = len(projected_failures) / len(utterances)
        if extra or failure_ratio > args.max_failed_ratio:
            raise RuntimeError(
                "MFA failure limit exceeded in chunk {}: expected={} produced={} "
                "missing={} extra={} corpus_failure_ratio={:.2%} limit={:.2%}".format(
                    chunk_id,
                    len(expected),
                    len(produced),
                    missing[:20],
                    extra[:20],
                    failure_ratio,
                    args.max_failed_ratio,
                )
            )
        failed_ids.update(missing)
        if missing:
            print(
                "[warn] MFA did not export {} utterances; recorded as explicit "
                "failures: {}".format(len(missing), missing[:20])
            )
        for textgrid in produced_paths:
            destination = output / textgrid.name
            shutil.copy2(textgrid, destination)

        marker.write_text(
            json.dumps(
                {
                    **chunk_spec,
                    "chunk_id": chunk_id,
                    "textgrid_count": len(produced),
                    "failed_ids": missing,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        shutil.rmtree(chunk_root, ignore_errors=True)
        metadata_path.write_text(
            json.dumps(
                {
                    **run_spec,
                    "run_id": run_id,
                    "status": "in_progress",
                    "failed_ids": sorted(failed_ids),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    actual = {path.stem for path in output.glob("*.TextGrid")}
    expected_all = {item["id"] for item in utterances}
    missing_all = sorted(expected_all - actual - failed_ids)
    extra_all = sorted(actual - expected_all)
    if missing_all or extra_all:
        raise RuntimeError(
            "Final alignment mismatch: missing={} extra={}".format(
                missing_all[:20], extra_all[:20]
            )
        )

    metadata = {
        **run_spec,
        "run_id": run_id,
        "textgrid_count": len(actual),
        "max_utterances": args.max_utterances,
        "status": "complete",
        "failed_ids": sorted(failed_ids),
        "failed_count": len(failed_ids),
        "failed_ratio": len(failed_ids) / len(expected_all),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[PASS] aligned {} utterances; run_id={}".format(len(actual), run_id))


if __name__ == "__main__":
    main()
