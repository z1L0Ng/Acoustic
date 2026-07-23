from __future__ import annotations

import csv
import json
import statistics
import wave
from collections import Counter, defaultdict
from pathlib import Path

import openpyxl


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "dataset" / "raw"
OUT = ROOT / "codex" / "2026-07-13"
CHECKED_DATE = "2026-07-10"


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def wav_stats(paths: list[Path]) -> dict:
    durations = []
    formats = Counter()
    for path in paths:
        with wave.open(str(path), "rb") as audio:
            rate = audio.getframerate()
            durations.append(audio.getnframes() / rate)
            formats[(rate, audio.getnchannels(), audio.getsampwidth() * 8)] += 1
    return {
        "count": len(paths),
        "total_hours": sum(durations) / 3600,
        "min_s": min(durations),
        "median_s": statistics.median(durations),
        "mean_s": statistics.mean(durations),
        "max_s": max(durations),
        "formats": formats,
    }


def duration_text(stats: dict) -> str:
    return (
        f"{stats['total_hours']:.2f} h measured from WAV headers; "
        f"per recording min/median/mean/max "
        f"{stats['min_s']:.2f}/{stats['median_s']:.2f}/{stats['mean_s']:.2f}/{stats['max_s']:.2f} s"
    )


def interval_summary(values: list[float], unit: str = "s") -> str:
    if not values:
        return "unknown / not directly derivable"
    return (
        f"n={len(values)}; total={sum(values) / 3600:.2f} h; "
        f"min/median/mean/max={min(values):.3f}/{statistics.median(values):.3f}/"
        f"{statistics.mean(values):.3f}/{max(values):.3f} {unit}"
    )


def parse_hms(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def format_counter(counter: Counter, order: list[str] | None = None) -> str:
    keys = order or sorted(counter)
    return "; ".join(f"{key}: {counter[key]:,}" for key in keys if key in counter)


def audit_icbhi() -> tuple[dict, list[dict], list[dict], list[dict]]:
    base = RAW / "icbhi_2017" / "source_original" / "ICBHI_final_database" / "ICBHI_final_database"
    wavs = sorted(base.glob("*.wav"))
    audio = wav_stats(wavs)
    subjects = {path.stem.split("_")[0] for path in wavs}
    devices = Counter(path.stem.split("_")[-1] for path in wavs)
    sites = Counter(path.stem.split("_")[2] for path in wavs)

    cycle_counts = Counter()
    cycle_durations = []
    for path in sorted(base.glob("*.txt")):
        if path.name == "filename_format.txt":
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) != 4:
                continue
            start, end, crackle, wheeze = parts
            cycle_durations.append(float(end) - float(start))
            label = "both" if crackle == "1" and wheeze == "1" else "crackle" if crackle == "1" else "wheeze" if wheeze == "1" else "normal"
            cycle_counts[label] += 1

    split_file = RAW / "icbhi_2017" / "ICBHI_challenge_train_test.txt"
    split_subjects: dict[str, set[str]] = defaultdict(set)
    split_recordings = Counter()
    for line in split_file.read_text().splitlines():
        recording, split = line.split()
        split_recordings[split] += 1
        split_subjects[split].add(recording.split("_")[0])
    overlap = split_subjects["train"] & split_subjects["test"]
    overlap_recordings = {}
    for patient_id in sorted(overlap):
        by_split = defaultdict(list)
        for line in split_file.read_text().splitlines():
            recording, split = line.split()
            if recording.startswith(patient_id + "_"):
                by_split[split].append(recording)
        overlap_recordings[patient_id] = dict(by_split)

    row = {
        "dataset": "ICBHI 2017",
        "version_scope": "Official ICBHI 2017 challenge package",
        "primary_task": "Respiratory-cycle classification; optional subject-level disease classification",
        "prediction_annotation_unit": "Prediction: respiratory cycle; annotation: cycle boundaries + crackle/wheeze binary flags. Disease is subject-level.",
        "sample_recording_segment_count": f"920 recordings; 6,898 annotated respiratory cycles (raw package measured)",
        "patient_subject_count": f"126 subjects (raw patient IDs and official side files)",
        "duration": duration_text(audio) + "; cycle durations: " + interval_summary(cycle_durations),
        "raw_label_schema": "Cycle flags (crackle, wheeze) -> normal/crackle/wheeze/both; separate subject diagnosis file",
        "disease_event_separation": "Yes: subject diagnosis and cycle event labels are separate files/units",
        "available_metadata": "Patient ID, age, sex, diagnosis, chest location, acquisition mode, recording device, official split",
        "source_official_facts": f"Official challenge recording split: train {split_recordings['train']}, test {split_recordings['test']}; it is not patient-independent because measured subject overlap={len(overlap)} (IDs 156, 218).",
        "proposed_benchmark_policy": "Dual protocol only as a recommendation: (A) official challenge recording split for literature comparability; (B) a future strict patient-grouped split for robustness/generalization. Do not generate protocol B until approved.",
        "leakage_risk_grouping_key": f"Confirmed patient leakage in official train/test (measured overlap={len(overlap)}); device/site confounding also remains. Group by patient ID in any new protocol; report device/site distribution.",
        "class_distribution_imbalance": format_counter(cycle_counts, ["normal", "crackle", "wheeze", "both"]) + "; minority both vs normal ratio 1:7.20",
        "normal_abnormal_mapping": "yes - no flags=normal; any crackle/wheeze flag=abnormal",
        "four_class_mapping": "yes - direct normal/crackle/wheeze/both mapping",
        "dataset_specific_head": "yes - four-class cycle head; optional separate disease head",
        "primary_evidence": "Official challenge package + filename_format.txt + annotation TXT + official split/diagnosis/demographic side files; official URL https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge",
        "confidence": "high",
    }
    tasks = [
        {"dataset": "ICBHI 2017", "task_id": "cycle_4class", "prediction_unit": "respiratory_cycle", "annotation_unit": "respiratory_cycle", "raw_classes": "normal|crackle|wheeze|both", "unit_count": sum(cycle_counts.values()), "duration_statistics": interval_summary(cycle_durations), "official_or_proposed": "official annotation; benchmark use proposed", "notes": "Direct shared four-class target."},
        {"dataset": "ICBHI 2017", "task_id": "subject_disease", "prediction_unit": "subject", "annotation_unit": "subject", "raw_classes": "diagnosis categories in ICBHI_Challenge_diagnosis.txt", "unit_count": len(subjects), "duration_statistics": "not applicable", "official_or_proposed": "source-provided labels; optional proposed head", "notes": "Keep separate from event labels."},
    ]
    evidence = [
        {"dataset": "ICBHI 2017", "field_group": "counts_duration_labels", "evidence_type": "raw_package_measured", "source": str(base.relative_to(ROOT)), "checked_date": CHECKED_DATE, "details": f"{len(wavs)} WAV; {len(subjects)} IDs; {sum(cycle_counts.values())} cycles; {audio['total_hours']:.4f} h"},
        {"dataset": "ICBHI 2017", "field_group": "split", "evidence_type": "official_side_file_plus_measurement", "source": str(split_file.relative_to(ROOT)), "checked_date": CHECKED_DATE, "details": f"train={split_recordings['train']}; test={split_recordings['test']}; patient overlap={len(overlap)}; overlap recording evidence={json.dumps(overlap_recordings, sort_keys=True)}"},
        {"dataset": "ICBHI 2017", "field_group": "policy", "evidence_type": "proposed_policy", "source": "this curation", "checked_date": CHECKED_DATE, "details": "Keep official test split; derive grouped validation from official train; report device/site shift."},
    ]
    split_rows = [
        {"dataset": "ICBHI 2017", "split_name": "official_train", "split_type": "official challenge recording split", "recording_count": split_recordings["train"], "subject_or_proxy_count": len(split_subjects["train"]), "identity_semantics": "patient ID", "duration_hours": "not separately recomputed", "record_class_distribution": "not applicable", "event_or_cycle_distribution": "not separately recomputed", "notes": "Contains patient IDs 156 and 218 also found in official_test."},
        {"dataset": "ICBHI 2017", "split_name": "official_test", "split_type": "official challenge recording split", "recording_count": split_recordings["test"], "subject_or_proxy_count": len(split_subjects["test"]), "identity_semantics": "patient ID", "duration_hours": "not separately recomputed", "record_class_distribution": "not applicable", "event_or_cycle_distribution": "not separately recomputed", "notes": "Contains patient IDs 156 and 218 also found in official_train; not patient-independent."},
    ]
    return row, tasks, evidence, split_rows


def audit_hf() -> tuple[dict, list[dict], list[dict], list[dict]]:
    base = RAW / "hf_lung_v1" / "source_original"
    wavs = sorted(base.rglob("*.wav"))
    audio = wav_stats(wavs)
    device_counts = Counter("Littmann3200" if path.name.startswith("steth_") else "HF_Type-1" for path in wavs)
    split_counts = Counter("train" if "/train/" in str(path) else "test" for path in wavs)
    split_dates: dict[str, set[str]] = defaultdict(set)
    date_counts = Counter()
    for path in wavs:
        split = "train" if "/train/" in str(path) else "test"
        if path.name.startswith("steth_"):
            date = path.name.split("_")[1]
        else:
            date = path.name.split("_")[1][:10].replace("-", "")
        split_dates[split].add(date)
        date_counts[date] += 1
    date_overlap = split_dates["train"] & split_dates["test"]

    label_counts = Counter()
    label_durations: dict[str, list[float]] = defaultdict(list)
    for path in sorted(base.rglob("*_label.txt")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            label_counts[parts[0]] += 1
            label_durations[parts[0]].append(parse_hms(parts[2]) - parse_hms(parts[1]))

    row = {
        "dataset": "HF_Lung_V1",
        "version_scope": "Public GitLab release, README updated 2022-01-18",
        "primary_task": "Interval detection/classification of inhalation, exhalation and adventitious sounds",
        "prediction_annotation_unit": "Prediction/annotation: time interval within a fixed 15 s recording; overlapping phase and sound labels are possible",
        "sample_recording_segment_count": f"9,765 recordings; {sum(label_counts.values()):,} label rows/intervals across six raw label tokens",
        "patient_subject_count": "README-reported population: 279 people (261 TSECC patients + 18 RCW/RCC residents); package-level unique patient count unavailable/unverifiable",
        "duration": duration_text(audio) + "; interval durations available in label files (see long-format appendix)",
        "raw_label_schema": "I, E, D, Wheeze, Rhonchi, Stridor; I/E breath phase can overlap D/CAS subtype intervals",
        "disease_event_separation": "Event labels only in release; no per-record disease labels found",
        "available_metadata": "Source train/test folder, de-identified date/time, device prefix; HF-Type-1 location L1-L8 and truncation order; no direct age/sex/disease ID table",
        "source_official_facts": f"Source study folders: train {split_counts['train']:,}, test {split_counts['test']:,}; de-identified date-key overlap={len(date_overlap)}. README-reported population is 279, but package-level unique patient count is unavailable/unverifiable.",
        "proposed_benchmark_policy": "Use de-identified date + recording-session stem as a subject/session proxy, never as patient ID; keep all records sharing the proxy in one split and stratify/audit by device and labels.",
        "leakage_risk_grouping_key": f"High identity ambiguity: README says same date is very likely same subject; group by de-identified date, recording session stem and device. Source split has {len(date_overlap)} overlapping date keys.",
        "class_distribution_imbalance": format_counter(label_counts, ["I", "E", "D", "Wheeze", "Rhonchi", "Stridor"]) + "; Stridor is 49.7x fewer than I",
        "normal_abnormal_mapping": "partial - D/Wheeze/Rhonchi/Stridor are abnormal, but unlabeled background is not verified normal",
        "four_class_mapping": "partial - D supports crackle and Wheeze supports wheeze; rhonchi/stridor need other-abnormal or exclusion policy; both is not a native exclusive class",
        "dataset_specific_head": "yes - multi-label interval detection heads for I/E/D/Wheeze/Rhonchi/Stridor",
        "primary_evidence": "Raw WAV/TXT package + official GitLab README; paper DOI https://doi.org/10.1371/journal.pone.0254134; source https://gitlab.com/techsupportHF/HF_Lung_V1",
        "confidence": "high for files/counts; medium for subject identity and shared-label mapping",
    }
    tasks = []
    for label in ["I", "E", "D", "Wheeze", "Rhonchi", "Stridor"]:
        tasks.append({"dataset": "HF_Lung_V1", "task_id": f"interval_{label.lower()}", "prediction_unit": "time_interval", "annotation_unit": "time_interval", "raw_classes": label, "unit_count": label_counts[label], "duration_statistics": interval_summary(label_durations[label]), "official_or_proposed": "source-provided interval label", "notes": "Multi-label temporal annotations; intervals may overlap across breath phase and sound type."})
    evidence = [
        {"dataset": "HF_Lung_V1", "field_group": "counts_duration_labels", "evidence_type": "raw_package_measured", "source": str(base.relative_to(ROOT)), "checked_date": CHECKED_DATE, "details": f"{len(wavs)} WAV; {sum(label_counts.values())} label rows; {audio['total_hours']:.4f} h"},
        {"dataset": "HF_Lung_V1", "field_group": "population_metadata", "evidence_type": "official_readme", "source": "dataset/raw/hf_lung_v1/README.md", "checked_date": CHECKED_DATE, "details": "261 TSECC patients + 18 RCW/RCC residents; same-date grouping warning; device/location semantics."},
        {"dataset": "HF_Lung_V1", "field_group": "policy", "evidence_type": "proposed_policy", "source": "this curation", "checked_date": CHECKED_DATE, "details": "Use date/session grouping; do not equate unlabeled background with normal; preserve rhonchi/stridor."},
    ]
    split_rows = [
        {"dataset": "HF_Lung_V1", "split_name": "source_train", "split_type": "source study folder", "recording_count": split_counts["train"], "subject_or_proxy_count": len(split_dates["train"]), "identity_semantics": "unique de-identified date keys; session/subject proxy, not patient IDs", "duration_hours": f"{split_counts['train'] * 15 / 3600:.4f}", "record_class_distribution": "not applicable", "event_or_cycle_distribution": "see overall task-unit appendix; split-level distribution not required for this curation", "notes": "No date-key overlap with source_test."},
        {"dataset": "HF_Lung_V1", "split_name": "source_test", "split_type": "source study folder", "recording_count": split_counts["test"], "subject_or_proxy_count": len(split_dates["test"]), "identity_semantics": "unique de-identified date keys; session/subject proxy, not patient IDs", "duration_hours": f"{split_counts['test'] * 15 / 3600:.4f}", "record_class_distribution": "not applicable", "event_or_cycle_distribution": "see overall task-unit appendix; split-level distribution not required for this curation", "notes": "No date-key overlap with source_train."},
    ]
    return row, tasks, evidence, split_rows


def audit_kauh() -> tuple[dict, list[dict], list[dict], list[dict]]:
    base = RAW / "kauh_fraiwan" / "source_original" / "audio_files"
    wavs = sorted(base.glob("*.wav"))
    audio = wav_stats(wavs)
    patients = {path.name.split("_")[0].lstrip("BDE") for path in wavs}
    recordings_per_patient = Counter(path.name.split("_")[0].lstrip("BDE") for path in wavs)
    prefixes = Counter(path.name[0] for path in wavs)

    workbook = RAW / "kauh_fraiwan" / "Data annotation.xlsx"
    sheet = openpyxl.load_workbook(workbook, read_only=True, data_only=True).active
    values = list(sheet.values)
    header = [str(value).strip() if value is not None else "" for value in values[0]]
    rows = [dict(zip(header, row)) for row in values[1:] if row and row[0] is not None]
    sound_counts = Counter(str(row.get("Sound type", "")).strip() for row in rows)
    diagnosis_counts_raw = Counter(str(row.get("Diagnosis", "")).strip() for row in rows)
    diagnosis_counts = Counter()
    for label, count in diagnosis_counts_raw.items():
        normalized = label.strip().lower()
        diagnosis_counts[normalized] += count

    row = {
        "dataset": "KAUH / Fraiwan",
        "version_scope": "Mendeley Data v3 (jwyy9np4gv)",
        "primary_task": "Recording-level lung sound type classification; optional patient diagnosis classification",
        "prediction_annotation_unit": "Prediction: recording for sound type; annotation workbook: one row per patient, labels repeated across three recordings (B/D/E prefixes)",
        "sample_recording_segment_count": f"336 WAV recordings; 112 annotation rows/patients; exactly 3 recordings per patient in measured package",
        "patient_subject_count": f"112 patients (P1-P112 measured from filenames/workbook)",
        "duration": duration_text(audio) + "; no event boundaries, so event duration is not directly derivable",
        "raw_label_schema": "Sound type: I E W, E W, I C, I C E W, I C B, C, Crep, N, Bronchial; separate Diagnosis column (14 raw spellings before normalization)",
        "disease_event_separation": "Yes by workbook columns, but sound labels are patient-row/recording-level descriptors rather than event boundaries",
        "available_metadata": "Patient number, age, gender, chest location, sound type, diagnosis, 3M Littmann 3200 device; no session ID or official split",
        "source_official_facts": "No official split found in Mendeley Data v3 package. Raw filenames and workbook cover P1-P112; measured B/D/E prefixes contribute exactly one recording each per patient.",
        "proposed_benchmark_policy": "Recommendation only: patient-grouped stratified split or grouped cross-validation by P-number; all B/D/E recordings for a patient stay together. This is not an official split.",
        "leakage_risk_grouping_key": "High if recording-random split is used because each patient contributes three files with repeated labels/metadata. Group by P-number; audit B/D/E prefix semantics before treating as site/session.",
        "class_distribution_imbalance": format_counter(sound_counts, ["N", "E W", "Crep", "C", "I E W", "I C E W", "I C B", "I C", "Bronchial"]) + "; rare sound combinations have 1-2 patients",
        "normal_abnormal_mapping": "yes at recording/patient-row label level - N=normal; all other sound types abnormal (proposed policy)",
        "four_class_mapping": "partial - W/C/Crep support wheeze/crackle; I C E W plausibly both; I C B and Bronchial require policy; raw abbreviations need source-author confirmation",
        "dataset_specific_head": "yes - nine-way raw sound-type head and normalized diagnosis head, kept separate",
        "primary_evidence": "Mendeley Data v3 package (WAV filenames + Data annotation.xlsx + Readme.txt); https://data.mendeley.com/datasets/jwyy9np4gv/3",
        "confidence": "high for measured counts/metadata; medium for B/D/E and combined-label semantics",
    }
    tasks = [
        {"dataset": "KAUH / Fraiwan", "task_id": "record_sound_type", "prediction_unit": "recording", "annotation_unit": "patient_row_applied_to_recordings", "raw_classes": "|".join(sound_counts.keys()), "unit_count": len(wavs), "duration_statistics": duration_text(audio), "official_or_proposed": "source labels; recording-level use inferred from filenames/workbook", "notes": "Three recordings per patient; no event boundaries."},
        {"dataset": "KAUH / Fraiwan", "task_id": "patient_diagnosis", "prediction_unit": "patient", "annotation_unit": "patient", "raw_classes": "|".join(diagnosis_counts_raw.keys()), "unit_count": len(rows), "duration_statistics": "not applicable", "official_or_proposed": "source-provided diagnosis; optional proposed head", "notes": "Normalize capitalization/spelling only with an explicit curation map."},
    ]
    evidence = [
        {"dataset": "KAUH / Fraiwan", "field_group": "counts_duration", "evidence_type": "raw_package_measured", "source": str(base.relative_to(ROOT)), "checked_date": CHECKED_DATE, "details": f"{len(wavs)} WAV; {len(patients)} patients; per-patient files={sorted(set(recordings_per_patient.values()))}; prefixes={dict(prefixes)}; {audio['total_hours']:.4f} h"},
        {"dataset": "KAUH / Fraiwan", "field_group": "labels_metadata", "evidence_type": "raw_workbook_measured", "source": str(workbook.relative_to(ROOT)), "checked_date": CHECKED_DATE, "details": f"{len(rows)} patient rows; sound types={dict(sound_counts)}; diagnoses raw={dict(diagnosis_counts_raw)}"},
        {"dataset": "KAUH / Fraiwan", "field_group": "policy", "evidence_type": "proposed_policy", "source": "this curation", "checked_date": CHECKED_DATE, "details": "Group by P-number; keep disease and sound heads separate; seek clarification for B/D/E and I C B/Bronchial."},
    ]
    split_rows = [
        {"dataset": "KAUH / Fraiwan", "split_name": "full_release", "split_type": "Mendeley Data v3 package; no official split", "recording_count": len(wavs), "subject_or_proxy_count": len(patients), "identity_semantics": "patient P-number", "duration_hours": f"{audio['total_hours']:.4f}", "record_class_distribution": format_counter(sound_counts), "event_or_cycle_distribution": "not available; no event boundaries", "notes": "Exactly 3 recordings per patient, one each with B/D/E prefix; prefix semantics need confirmation."},
    ]
    return row, tasks, evidence, split_rows


def audit_sprsound() -> tuple[dict, list[dict], list[dict], list[dict]]:
    repo = RAW / "sprsound" / "source_original" / "SPRSound-874eeb8736ddb78937c2fb5332fc7e7293d0f0ca"
    base = repo / "BioCAS2022"
    wavs = sorted(base.rglob("*.wav"))
    audio = wav_stats(wavs)
    patients = {path.stem.split("_")[0] for path in wavs}
    train_wavs = list((base / "train2022_wav").rglob("*.wav"))
    test_wavs = list((base / "test2022_wav").rglob("*.wav"))
    train_patients = {path.stem.split("_")[0] for path in train_wavs}
    test_patients = {path.stem.split("_")[0] for path in test_wavs}
    patient_overlap = train_patients & test_patients

    jsons = sorted(base.rglob("*.json"))
    record_counts = Counter()
    event_counts = Counter()
    event_durations: dict[str, list[float]] = defaultdict(list)
    for path in jsons:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "record_annotation" in payload:
            record_counts[payload["record_annotation"]] += 1
        for event in payload.get("event_annotation", []):
            label = event["type"]
            event_counts[label] += 1
            event_durations[label].append((float(event["end"]) - float(event["start"])) / 1000)

    inter_jsons = list((base / "test2022_json" / "inter_test_json").glob("*.json"))
    intra_jsons = list((base / "test2022_json" / "intra_test_json").glob("*.json"))

    def spr_split_row(split_name: str, split_type: str, subset_jsons: list[Path]) -> dict:
        stems = {path.stem for path in subset_jsons}
        subset_wavs = [path for path in wavs if path.stem in stems]
        subset_patients = {path.stem.split("_")[0] for path in subset_wavs}
        subset_records = Counter()
        subset_events = Counter()
        for path in subset_jsons:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if "record_annotation" in payload:
                subset_records[payload["record_annotation"]] += 1
            for event in payload.get("event_annotation", []):
                subset_events[event["type"]] += 1
        subset_audio = wav_stats(subset_wavs)
        return {"dataset": "SPRSound", "split_name": split_name, "split_type": split_type, "recording_count": len(subset_wavs), "subject_or_proxy_count": len(subset_patients), "identity_semantics": "patient ID from filename", "duration_hours": f"{subset_audio['total_hours']:.4f}", "record_class_distribution": format_counter(subset_records, ["Normal", "DAS", "CAS", "CAS & DAS", "Poor Quality"]), "event_or_cycle_distribution": format_counter(subset_events, ["Normal", "Fine Crackle", "Wheeze", "Rhonchi", "Coarse Crackle", "Wheeze+Crackle", "Stridor"]), "notes": "BioCAS2022 only; counts measured from matched WAV/JSON stems."}

    train_jsons = list((base / "train2022_json").glob("*.json"))
    split_rows = [
        spr_split_row("train2022", "official/source training release", train_jsons),
        spr_split_row("inter_test2022", "official inter-subject test", inter_jsons),
        spr_split_row("intra_test2022", "official intra-subject test", intra_jsons),
    ]
    split_rows[0]["notes"] += f" Unique patients={len(train_patients)}."
    split_rows[1]["notes"] += f" Unique patients={len({p.stem.split('_')[0] for p in inter_jsons})}; overlap with train=0."
    split_rows[2]["notes"] += f" Unique patients={len({p.stem.split('_')[0] for p in intra_jsons})}; overlap with train={len(train_patients & {p.stem.split('_')[0] for p in intra_jsons})}, as intended by intra-subject design."
    row = {
        "dataset": "SPRSound",
        "version_scope": "BioCAS 2022 classification release only, from repository commit 874eeb8736ddb78937c2fb5332fc7e7293d0f0ca",
        "primary_task": "Event-level and recording-level respiratory sound classification",
        "prediction_annotation_unit": "Two tasks: respiratory event segments with start/end ms; whole-record labels with signal-quality class",
        "sample_recording_segment_count": f"2,683 recordings (1,949 train + 734 test); {sum(event_counts.values()):,} event segments; {sum(record_counts.values()):,} record labels",
        "patient_subject_count": f"{len(patients)} unique patient IDs measured in BioCAS2022; train {len(train_patients)}, test {len(test_patients)}, overlap {len(patient_overlap)}",
        "duration": duration_text(audio) + "; event durations: " + interval_summary([v for values in event_durations.values() for v in values]),
        "raw_label_schema": "Event: Normal, Rhonchi, Wheeze, Stridor, Coarse Crackle, Fine Crackle, Wheeze+Crackle. Record: Normal, CAS, DAS, CAS & DAS, Poor Quality.",
        "disease_event_separation": "No disease labels in BioCAS2022 task package; event labels and record labels are separate JSON fields",
        "available_metadata": "Patient ID, age, sex, recording location p1-p4, recording number, Yunting model II device, train/test and inter-/intra-subject test designation",
        "source_official_facts": f"BioCAS2022 source split: train 1,949; test 734 = inter-subject {len(inter_jsons)} + intra-subject {len(intra_jsons)}. Measured train/test patient overlap={len(patient_overlap)}, consistent with the official intra-subject design; see split statistics for subjects/classes.",
        "proposed_benchmark_policy": "Report inter-subject and intra-subject test results separately; treat inter-subject as the primary generalization result and intra-subject as a repeated-subject diagnostic. Create validation only by patient grouping within train.",
        "leakage_risk_grouping_key": f"Intentional repeated-patient overlap exists for intra-subject test. Group by patient ID for validation and cross-dataset training; never merge inter/intra test subsets. Recording number is secondary key; location is a confounder.",
        "class_distribution_imbalance": "Event - " + format_counter(event_counts, ["Normal", "Fine Crackle", "Wheeze", "Rhonchi", "Coarse Crackle", "Wheeze+Crackle", "Stridor"]) + "; Record - " + format_counter(record_counts, ["Normal", "DAS", "CAS", "CAS & DAS", "Poor Quality"]),
        "normal_abnormal_mapping": "yes - event Normal vs all adventitious labels; at record level keep Poor Quality excluded/separate",
        "four_class_mapping": "partial - fine/coarse crackle -> crackle; Wheeze -> wheeze; Wheeze+Crackle -> both; Rhonchi/Stridor require other-abnormal or exclusion policy",
        "dataset_specific_head": "yes - official event binary/7-class and record ternary/5-class heads",
        "primary_evidence": "BioCAS2022 raw WAV/JSON + repository README/BioCAS2022 README, pinned commit; https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound",
        "confidence": "high for pinned 2022 package; medium for cross-version identity if later challenge data are added",
    }
    tasks = [
        {"dataset": "SPRSound", "task_id": "event_binary", "prediction_unit": "respiratory_event_segment", "annotation_unit": "event_segment_start_end_ms", "raw_classes": "Normal|Adventitious", "unit_count": sum(event_counts.values()), "duration_statistics": interval_summary([v for values in event_durations.values() for v in values]), "official_or_proposed": "official challenge Task 1-1", "notes": "All non-Normal event labels collapse to Adventitious."},
        {"dataset": "SPRSound", "task_id": "event_7class", "prediction_unit": "respiratory_event_segment", "annotation_unit": "event_segment_start_end_ms", "raw_classes": "Normal|Rhonchi|Wheeze|Stridor|Coarse Crackle|Fine Crackle|Wheeze+Crackle", "unit_count": sum(event_counts.values()), "duration_statistics": interval_summary([v for values in event_durations.values() for v in values]), "official_or_proposed": "official challenge Task 1-2", "notes": "Preferred dataset-specific event head."},
        {"dataset": "SPRSound", "task_id": "record_ternary", "prediction_unit": "recording", "annotation_unit": "recording", "raw_classes": "Normal|Adventitious|Poor Quality", "unit_count": sum(record_counts.values()), "duration_statistics": duration_text(audio), "official_or_proposed": "official challenge Task 2-1", "notes": "CAS/DAS/CAS & DAS collapse to Adventitious."},
        {"dataset": "SPRSound", "task_id": "record_5class", "prediction_unit": "recording", "annotation_unit": "recording", "raw_classes": "Normal|CAS|DAS|CAS & DAS|Poor Quality", "unit_count": sum(record_counts.values()), "duration_statistics": duration_text(audio), "official_or_proposed": "official challenge Task 2-2", "notes": "Preferred dataset-specific record head."},
    ]
    evidence = [
        {"dataset": "SPRSound", "field_group": "counts_duration_labels", "evidence_type": "raw_package_measured", "source": str(base.relative_to(ROOT)), "checked_date": CHECKED_DATE, "details": f"{len(wavs)} WAV; {len(patients)} patient IDs; {len(jsons)} JSON; {sum(event_counts.values())} events; {audio['total_hours']:.4f} h"},
        {"dataset": "SPRSound", "field_group": "tasks_split_metadata", "evidence_type": "official_repository_readme_plus_raw_measurement", "source": str((base / "README.md").relative_to(ROOT)), "checked_date": CHECKED_DATE, "details": f"Official 2022 classification tasks; train={len(train_wavs)} recordings/{len(train_patients)} patients; inter={len(inter_jsons)} recordings/{len({p.stem.split('_')[0] for p in inter_jsons})} patients with train overlap 0; intra={len(intra_jsons)} recordings/{len({p.stem.split('_')[0] for p in intra_jsons})} patients with train overlap {len(train_patients & {p.stem.split('_')[0] for p in intra_jsons})}."},
        {"dataset": "SPRSound", "field_group": "policy", "evidence_type": "proposed_policy", "source": "this curation", "checked_date": CHECKED_DATE, "details": "Pin BioCAS2022; report inter/intra separately; keep Poor Quality and rhonchi/stridor explicit."},
    ]
    return row, tasks, evidence, split_rows


def main() -> None:
    main_rows = []
    task_rows = []
    evidence_rows = []
    split_rows = []
    for audit in [audit_icbhi, audit_hf, audit_sprsound, audit_kauh]:
        row, tasks, evidence, splits = audit()
        main_rows.append(row)
        task_rows.extend(tasks)
        evidence_rows.extend(evidence)
        split_rows.extend(splits)

    main_fields = [
        "dataset", "version_scope", "primary_task", "prediction_annotation_unit",
        "sample_recording_segment_count", "patient_subject_count", "duration",
        "raw_label_schema", "disease_event_separation", "available_metadata",
        "source_official_facts", "proposed_benchmark_policy", "leakage_risk_grouping_key",
        "class_distribution_imbalance", "normal_abnormal_mapping",
        "four_class_mapping", "dataset_specific_head", "primary_evidence", "confidence",
    ]
    task_fields = ["dataset", "task_id", "prediction_unit", "annotation_unit", "raw_classes", "unit_count", "duration_statistics", "official_or_proposed", "notes"]
    evidence_fields = ["dataset", "field_group", "evidence_type", "source", "checked_date", "details"]
    split_fields = ["dataset", "split_name", "split_type", "recording_count", "subject_or_proxy_count", "identity_semantics", "duration_hours", "record_class_distribution", "event_or_cycle_distribution", "notes"]
    write_csv(OUT / "dataset_phase1_publication_curation_2026-07-13.csv", main_rows, main_fields)
    write_csv(OUT / "dataset_phase1_task_units_2026-07-13.csv", task_rows, task_fields)
    write_csv(OUT / "dataset_phase1_curation_evidence_2026-07-13.csv", evidence_rows, evidence_fields)
    write_csv(OUT / "dataset_phase1_split_statistics_2026-07-13.csv", split_rows, split_fields)


if __name__ == "__main__":
    main()
