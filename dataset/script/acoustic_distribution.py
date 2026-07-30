"""Four-dataset acoustic distribution analysis (read-only over raw audio).

Computes per-recording acoustic descriptors for ICBHI 2017, SPRSound
BioCAS2022, HF_Lung_V1 and KAUH/Fraiwan, then aggregates them into
dataset / device / class summaries plus a trivial dataset-identity probe.

Design notes
------------
* Identity, device, site, split and class parsing reuse the canonical logic
  in ``dataset/script/publication_curation_audit.py`` so numbers stay
  consistent with the maintained curation.
* Amplitude / SNR descriptors are computed at each file's native sample rate.
* Spectral descriptors are computed after resampling every file to a common
  ``ANALYSIS_SR`` (4 kHz) so cross-dataset spectra are compared on the same
  frequency grid. All four datasets are natively >= 4 kHz, and respiratory
  diagnostic content is < 2 kHz, so this is loss-free for the band of interest.
* The estimated SNR is an energy-percentile proxy (loud-frame dB minus
  quiet-frame dB), explicitly a proxy, not a calibrated measurement.

Outputs land under ``result/acoustic_distribution/`` (git-ignored); the script
and the companion notebook are the committed, reproducible artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "dataset" / "raw"
OUT = ROOT / "result" / "acoustic_distribution"

ANALYSIS_SR = 4000  # common analysis rate for spectral descriptors
FRAME_MS = 25.0
HOP_MS = 10.0
BANDS_HZ = [(0, 100), (100, 250), (250, 500), (500, 1000), (1000, 2000)]


# --------------------------------------------------------------------------- #
# Recording enumeration (canonical identity/label parsing per dataset)
# --------------------------------------------------------------------------- #
def _icbhi_records() -> list[dict]:
    base = RAW / "icbhi_2017/source_original/ICBHI_final_database/ICBHI_final_database"
    split_file = RAW / "icbhi_2017/ICBHI_challenge_train_test.txt"
    split = {}
    for line in split_file.read_text().splitlines():
        rec, part = line.split()
        split[rec] = part
    rows = []
    for wav in sorted(base.glob("*.wav")):
        stem = wav.stem
        parts = stem.split("_")
        # cycle labels -> recording-level abnormal flag
        txt = wav.with_suffix(".txt")
        klass = "unlabeled"
        if txt.is_file():
            any_abn = False
            n_cycles = 0
            for line in txt.read_text(errors="replace").splitlines():
                p = line.split()
                if len(p) != 4:
                    continue
                n_cycles += 1
                if p[2] == "1" or p[3] == "1":
                    any_abn = True
            klass = "abnormal" if any_abn else ("normal" if n_cycles else "unlabeled")
        rows.append(
            {
                "dataset": "ICBHI",
                "filename": wav.name,
                "path": str(wav),
                "patient_id": parts[0],
                "device": parts[-1],
                "site": parts[2] if len(parts) > 2 else "NA",
                "split": split.get(stem, "NA"),
                "recording_class": klass,
            }
        )
    return rows


def _sprsound_records() -> list[dict]:
    base = RAW / "sprsound/source_original/SPRSound-874eeb8736ddb78937c2fb5332fc7e7293d0f0ca/BioCAS2022"
    # map recording stem -> record_annotation and inter/intra membership from JSON tree
    rec_label = {}
    inter = set()
    intra = set()
    for jp in base.rglob("*.json"):
        try:
            payload = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        stem = jp.stem
        if "record_annotation" in payload:
            rec_label[stem] = payload["record_annotation"]
        if "inter_test_json" in str(jp):
            inter.add(stem)
        elif "intra_test_json" in str(jp):
            intra.add(stem)
    rows = []
    for wav in sorted(base.rglob("*.wav")):
        stem = wav.stem
        part = "train"
        if stem in inter:
            part = "inter_test"
        elif stem in intra:
            part = "intra_test"
        rows.append(
            {
                "dataset": "SPRSound",
                "filename": wav.name,
                "path": str(wav),
                "patient_id": stem.split("_")[0],
                "device": "Yunting_II",
                "site": stem.split("_")[3] if len(stem.split("_")) > 3 else "NA",
                "split": part,
                "recording_class": rec_label.get(stem, "NA"),
            }
        )
    return rows


def _hf_records() -> list[dict]:
    base = RAW / "hf_lung_v1/source_original"
    rows = []
    for wav in sorted(base.rglob("*.wav")):
        name = wav.name
        steth = name.startswith("steth_")
        device = "Littmann3200" if steth else "HF_Type-1"
        split = "train" if "/train/" in str(wav) else "test"
        label_file = wav.with_name(wav.stem + "_label.txt")
        klass = "unlabeled"
        if label_file.is_file():
            tokens = set()
            for line in label_file.read_text(errors="replace").splitlines():
                p = line.split()
                if p:
                    tokens.add(p[0])
            adv = tokens & {"D", "Wheeze", "Rhonchi", "Stridor"}
            klass = "adventitious" if adv else ("phase_only" if tokens else "unlabeled")
        date = name.split("_")[1] if steth else name.split("_")[1][:10].replace("-", "")
        rows.append(
            {
                "dataset": "HF_Lung",
                "filename": name,
                "path": str(wav),
                "patient_id": f"date:{date}",  # proxy identity only
                "device": device,
                "site": "NA",
                "split": split,
                "recording_class": klass,
            }
        )
    return rows


def _kauh_records() -> list[dict]:
    base = RAW / "kauh_fraiwan/source_original/audio_files"
    rows = []
    for wav in sorted(base.glob("*.wav")):
        name = wav.name
        fields = name.split(",")  # BP100_N , N , P R M , 70 , F
        diagnosis = fields[0].split("_", 1)[1] if "_" in fields[0] else "NA"
        sound = fields[1].strip() if len(fields) > 1 else "NA"
        prefix = name[0]  # B/D/E
        pnum = fields[0].split("_")[0].lstrip("BDE")
        rows.append(
            {
                "dataset": "KAUH",
                "filename": name,
                "path": str(wav),
                "patient_id": pnum,
                "device": "Littmann3200",
                "site": fields[2].strip() if len(fields) > 2 else "NA",
                "split": f"prefix_{prefix}",
                "recording_class": "normal" if sound == "N" else "abnormal",
                "sound_type": sound,
                "diagnosis": diagnosis,
            }
        )
    return rows


def enumerate_records() -> list[dict]:
    rows = []
    for fn in (_icbhi_records, _sprsound_records, _hf_records, _kauh_records):
        rows.extend(fn())
    return rows


# --------------------------------------------------------------------------- #
# Acoustic feature extraction
# --------------------------------------------------------------------------- #
def _frame_db(y: np.ndarray, sr: int) -> np.ndarray:
    win = max(1, int(sr * FRAME_MS / 1000))
    hop = max(1, int(sr * HOP_MS / 1000))
    if len(y) < win:
        return np.array([20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-12)])
    n = 1 + (len(y) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = y[idx]
    energy = np.sqrt(np.mean(frames**2, axis=1)) + 1e-12
    return 20 * np.log10(energy)


def extract_features(path: str, native_sr_hint: int | None = None) -> dict | None:
    try:
        y, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        return None
    if y.ndim > 1:
        y = y.mean(axis=1)
    if y.size == 0:
        return None
    n = y.size
    dc = float(np.mean(y))
    y = y - dc
    rms = float(np.sqrt(np.mean(y**2)) + 1e-12)
    peak = float(np.max(np.abs(y)) + 1e-12)
    dbfs = 20 * np.log10(rms)
    crest_db = 20 * np.log10(peak / rms)
    clip_frac = float(np.mean(np.abs(y) > 0.99))

    fdb = _frame_db(y, sr)
    snr_db = float(np.percentile(fdb, 90) - np.percentile(fdb, 10))

    # spectral descriptors at common analysis rate
    if sr != ANALYSIS_SR:
        g = np.gcd(int(sr), ANALYSIS_SR)
        ya = resample_poly(y, ANALYSIS_SR // g, sr // g).astype(np.float32)
    else:
        ya = y
    if ya.size < 256:
        ya = np.pad(ya, (0, 256 - ya.size))
    win = np.hanning(min(1024, len(ya)))
    seg = ya[: len(win)] if len(ya) >= len(win) else np.pad(ya, (0, len(win) - len(ya)))
    # Welch-like average over the whole file
    nfft = 1024
    step = nfft // 2
    mags = []
    w = np.hanning(nfft)
    for start in range(0, max(1, len(ya) - nfft + 1), step):
        frame = ya[start : start + nfft]
        if len(frame) < nfft:
            break
        spec = np.abs(np.fft.rfft(frame * w)) ** 2
        mags.append(spec)
    if not mags:
        frame = np.pad(ya[:nfft], (0, max(0, nfft - len(ya))))
        mags = [np.abs(np.fft.rfft(frame * w)) ** 2]
    psd = np.mean(mags, axis=0)
    freqs = np.fft.rfftfreq(nfft, 1.0 / ANALYSIS_SR)
    psd_sum = float(psd.sum()) + 1e-12
    centroid = float(np.sum(freqs * psd) / psd_sum)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / psd_sum))
    cumulative = np.cumsum(psd) / psd_sum
    rolloff = float(freqs[np.searchsorted(cumulative, 0.85)])
    geo = np.exp(np.mean(np.log(psd + 1e-12)))
    flatness = float(geo / (np.mean(psd) + 1e-12))
    band_frac = {}
    for lo, hi in BANDS_HZ:
        m = (freqs >= lo) & (freqs < hi)
        band_frac[f"band_{lo}_{hi}"] = float(psd[m].sum() / psd_sum)
    zcr = float(np.mean(np.abs(np.diff(np.sign(ya))) > 0) )

    return {
        "native_sr": int(sr),
        "n_samples": int(n),
        "duration_s": n / sr,
        "dc_offset": dc,
        "rms": rms,
        "dbfs": dbfs,
        "crest_db": crest_db,
        "clip_frac": clip_frac,
        "snr_proxy_db": snr_db,
        "centroid_hz": centroid,
        "bandwidth_hz": bandwidth,
        "rolloff85_hz": rolloff,
        "flatness": flatness,
        "zcr": zcr,
        **band_frac,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap files PER dataset (0 = all)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    records = enumerate_records()
    per_ds = Counter(r["dataset"] for r in records)
    print("enumerated recordings:", dict(per_ds), "total", len(records))

    if args.limit:
        capped, seen = [], Counter()
        for r in records:
            if seen[r["dataset"]] < args.limit:
                capped.append(r)
                seen[r["dataset"]] += 1
        records = capped
        print("limited to", dict(Counter(r["dataset"] for r in records)))

    rows = []
    t0 = time.time()
    errors = 0
    for i, r in enumerate(records):
        feats = extract_features(r["path"])
        if feats is None:
            errors += 1
            continue
        rows.append({**r, **feats})
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(records)} ({time.time() - t0:.0f}s, {errors} errors)")
    print(f"done {len(rows)} ok, {errors} errors, {time.time() - t0:.0f}s")

    # union of keys across datasets (KAUH adds sound_type/diagnosis)
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with (args.out / "recording_features.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out / "recording_features.csv")


if __name__ == "__main__":
    main()
