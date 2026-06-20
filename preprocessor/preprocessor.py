import os
import random
import json
import shutil
import unicodedata

import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]

        assert config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        self.pitch_phoneme_averaging = (
            config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )

        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def build_from_path(self):
        feature_dirs = ["mel", "pitch", "energy", "duration"]
        if self.config["preprocessing"].get("clean_output", True):
            for name in feature_dirs:
                shutil.rmtree(os.path.join(self.out_dir, name), ignore_errors=True)
        for name in feature_dirs:
            os.makedirs(os.path.join(self.out_dir, name), exist_ok=True)

        self._validate_alignment_coverage()

        print("Processing Data ...")
        out = list()
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()
        feature_rejected = []

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}
        speaker_names = sorted(
            name
            for name in os.listdir(self.in_dir)
            if os.path.isdir(os.path.join(self.in_dir, name))
        )
        for i, speaker in enumerate(tqdm(speaker_names)):
            speakers[speaker] = i
            # Archive extraction order differs between filesystems/runtimes.
            # Sort before the seeded shuffle below so train/val membership is
            # reproducible locally and in Colab.
            for wav_name in sorted(os.listdir(os.path.join(self.in_dir, speaker))):
                if ".wav" not in wav_name:
                    continue

                basename = wav_name.split(".")[0]
                if basename in self.excluded_alignment_ids:
                    continue
                tg_path = os.path.join(
                    self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
                )
                if os.path.exists(tg_path):
                    ret = self.process_utterance(speaker, basename)
                    if ret is None:
                        feature_rejected.append(
                            {
                                "id": basename,
                                "speaker": speaker,
                                "reason": self.last_rejection_reason,
                            }
                        )
                        continue
                    else:
                        info, pitch, energy, n = ret
                    out.append(info)
                else:
                    # Coverage is checked before the loop; this protects against
                    # a file disappearing during preprocessing.
                    raise FileNotFoundError(tg_path)

                if len(pitch) > 0:
                    pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                if len(energy) > 0:
                    energy_scaler.partial_fit(energy.reshape((-1, 1)))
                n_frames += n

        feature_input_count = len(out) + len(feature_rejected)
        feature_rejected_ratio = (
            len(feature_rejected) / feature_input_count if feature_input_count else 1.0
        )
        feature_report = {
            "input_count": feature_input_count,
            "accepted_count": len(out),
            "rejected_count": len(feature_rejected),
            "rejected_ratio": feature_rejected_ratio,
            "rejected": feature_rejected,
        }
        feature_report_path = os.path.join(self.out_dir, "feature_extraction_report.json")
        with open(feature_report_path, "w", encoding="utf-8") as handle:
            json.dump(feature_report, handle, ensure_ascii=False, indent=2)
        maximum_feature_rejected_ratio = float(
            self.config["preprocessing"].get("max_feature_rejected_ratio", 0.01)
        )
        if (
            not out
            or feature_rejected_ratio > maximum_feature_rejected_ratio
        ):
            raise RuntimeError(
                "Feature extraction rejected {}/{} utterances ({:.2%}, allowed "
                "{:.2%}). Inspect {}.".format(
                    len(feature_rejected),
                    feature_input_count,
                    feature_rejected_ratio,
                    maximum_feature_rejected_ratio,
                    feature_report_path,
                )
            )

        print("Computing statistic quantities ...")
        # Perform normalization if necessary
        if self.pitch_normalization:
            pitch_mean = pitch_scaler.mean_[0]
            pitch_std = pitch_scaler.scale_[0]
        else:
            # A numerical trick to avoid normalization...
            pitch_mean = 0
            pitch_std = 1
        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1

        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy"), energy_mean, energy_std
        )

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        random.Random(self.config["preprocessing"].get("split_seed", 42)).shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size :]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out

    def _validate_alignment_coverage(self):
        """Fail before feature extraction if wav/lab/TextGrid sets differ."""
        raw = set()
        labs = set()
        grids = set()
        self.excluded_alignment_ids = set()
        run_metadata_path = os.path.join(
            self.out_dir, "TextGrid", "alignment_run.json"
        )
        if os.path.isfile(run_metadata_path):
            with open(run_metadata_path, encoding="utf-8") as handle:
                self.excluded_alignment_ids = set(
                    json.load(handle).get("failed_ids", [])
                )
        for speaker in os.listdir(self.in_dir):
            speaker_dir = os.path.join(self.in_dir, speaker)
            if not os.path.isdir(speaker_dir):
                continue
            raw.update(
                (speaker, os.path.splitext(name)[0])
                for name in os.listdir(speaker_dir)
                if name.endswith(".wav")
            )
            labs.update(
                (speaker, os.path.splitext(name)[0])
                for name in os.listdir(speaker_dir)
                if name.endswith(".lab")
            )
            tg_dir = os.path.join(self.out_dir, "TextGrid", speaker)
            if os.path.isdir(tg_dir):
                grids.update(
                    (speaker, os.path.splitext(name)[0])
                    for name in os.listdir(tg_dir)
                    if name.endswith(".TextGrid")
                )

        missing_labs = sorted(raw - labs)
        missing_grids = sorted(raw - grids)
        extra_grids = sorted(grids - raw)
        missing_grid_ids = {basename for _, basename in missing_grids}
        if (
            missing_labs
            or extra_grids
            or missing_grid_ids != self.excluded_alignment_ids
        ):
            raise RuntimeError(
                "Alignment coverage mismatch: wavs={}, labs={}, TextGrids={}, "
                "missing_labs={} missing_TextGrids={} extra_TextGrids={}. "
                "Run tools/validate_mfa_pipeline.py --stage alignment first."
                .format(
                    len(raw),
                    len(labs),
                    len(grids),
                    missing_labs[:10],
                    missing_grids[:10],
                    extra_grids[:10],
                )
            )

    def process_utterance(self, speaker, basename):
        self.last_rejection_reason = None
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))
        tg_path = os.path.join(
            self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
        )

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        text = "{" + " ".join(phone) + "}"
        if start >= end:
            self.last_rejection_reason = "empty_or_invalid_phone_alignment"
            return None

        # Read and trim wav files
        wav, _ = librosa.load(wav_path)
        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32)

        # Read raw text
        with open(text_path, "r", encoding="utf-8") as f:
            raw_text = f.readline().strip("\n")

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        pitch = pitch[: sum(duration)]
        if np.sum(pitch != 0) <= 1:
            self.last_rejection_reason = "insufficient_voiced_pitch"
            return None

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]

        if self.pitch_phoneme_averaging:
            # perform linear interpolation
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos : pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(duration)]

        if self.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos : pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]

        # Save files
        dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

        pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

        energy_filename = "{}-energy-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        return (
            "|".join([basename, speaker, text, raw_text]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
        )

    def get_alignment(self, tier):
        phones = []
        durations = []
        start_time = None
        end_time = None
        end_idx = 0

        for interval in tier._objects:
            start, end = interval.start_time, interval.end_time
            phone = unicodedata.normalize("NFC", interval.text.strip())

            # MFA 3 exports silence as an empty phone label.  Canonicalize both
            # empty and explicit silence to the single internal pause token.
            if phone in {"", "sil"}:
                phone = "sp"

            duration = int(
                np.round(end * self.sampling_rate / self.hop_length)
                - np.round(start * self.sampling_rate / self.hop_length)
            )

            # Trim leading silence.  ``spn`` is deliberately not silence: after
            # G2P-backed alignment it represents real unknown/noise and must stay
            # visible to validation and training.
            if not phones and phone == "sp":
                continue
            if start_time is None:
                start_time = start

            if phone == "sp" and phones and phones[-1] == "sp":
                durations[-1] += duration
            else:
                phones.append(phone)
                durations.append(duration)

            if phone != "sp":
                end_time = end
                end_idx = len(phones)

        phones = phones[:end_idx]
        durations = durations[:end_idx]
        if not phones or start_time is None or end_time is None:
            return [], [], 0, 0
        if len(phones) != len(durations) or any(not phone for phone in phones):
            raise RuntimeError("Invalid phone/duration alignment")
        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value
