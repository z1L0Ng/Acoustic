from __future__ import annotations

import builtins
import hashlib
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import torch

from baseline.multidataset_pipeline.beats_temporal import HFRawInterval
from baseline.multidataset_pipeline.contracts import ObservationState
from baseline.multidataset_pipeline.hf_data import (
    ASSIGNMENT_DIGEST_SERIALIZATION,
    HFExpectedCounts,
    HFSampleRecord,
    OWN_LABEL_TREE_DIGEST_SERIALIZATION,
    audit_wave_headers,
    build_hf_manifest,
    canonical_date_proxy,
    load_hf_waveform,
    parse_hms,
    parse_label_file,
    records_to_p6_supervision,
)


class HFDataContractTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.rows = [
            (
                "train/train/trunc_2020-01-01-00-00-00-L1_0",
                ["I 00:00:00.000 00:00:00.100"],
            ),
            (
                "train/train/trunc_2020-01-02-00-00-00-L1_0",
                ["E 00:00:00.100 00:00:00.200"],
            ),
            (
                "train/train/trunc_2020-01-03-00-00-00-L1_0",
                ["D 00:00:00.200 00:00:00.300"],
            ),
            (
                "train/train/trunc_2020-01-04-00-00-00-L1_0",
                ["Wheeze 00:00:00.300 00:00:00.400"],
            ),
            (
                "train/train/trunc_2020-01-05-00-00-00-L1_0",
                ["Rhonchi 00:00:00.400 00:00:00.500"],
            ),
            ("test/test/steth_20210101_00_00_00", []),
        ]
        for stem, labels in self.rows:
            path = self.root / f"{stem}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            self._write_wav(path)
            path.with_name(path.stem + "_label.txt").write_text(
                "\n".join(labels) + ("\n" if labels else ""),
                encoding="utf-8",
            )
        self.expected = HFExpectedCounts(
            recordings=6,
            source_split={"train": 5, "test": 1},
            raw_intervals=5,
            empty_label_files=1,
            token_counts={"I": 1, "E": 1, "D": 1, "Wheeze": 1, "Rhonchi": 1},
        )

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _write_wav(path: Path) -> None:
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(4_000)
            handle.writeframes(b"\x00\x00" * 60_000)

    def test_manifest_pairing_counts_empty_and_proxy_contract(self):
        records, receipt = build_hf_manifest(
            self.root, expected=self.expected
        )
        self.assertEqual(len(records), 6)
        self.assertEqual(receipt["raw_intervals"], 5)
        self.assertEqual(receipt["empty_label_files"], 1)
        self.assertEqual(receipt["explicit_negative_intervals"], 0)
        self.assertEqual(
            receipt["gap_semantics"],
            "not_annotated; never raw negative or shared normal",
        )
        empty = next(
            record
            for record in records
            if record.recording_state is ObservationState.EMPTY
        )
        self.assertEqual(empty.date_proxy, "2021-01-01")
        self.assertEqual(empty.group_id, "date_proxy:2021-01-01")
        self.assertFalse(Path(records[0].wav_relative_path).is_absolute())
        self.assertEqual(
            receipt["label_tree_identity_status"],
            "reference_not_reproduced_serialization_unknown",
        )
        self.assertEqual(
            receipt["own_label_tree_digest_serialization"],
            OWN_LABEL_TREE_DIGEST_SERIALIZATION,
        )

    def test_assignment_serialization_is_independently_reproducible(self):
        records, receipt = build_hf_manifest(
            self.root, expected=self.expected
        )
        rows = [
            f"{record.source_split}:{record.sample_id.split(':', 2)[2]}"
            f"\t{record.partition}\t{record.date_proxy.replace('-', '')}"
            for record in records
        ]
        digest = hashlib.sha256(
            ("\n".join(rows) + "\n").encode("utf-8")
        ).hexdigest()
        self.assertEqual(receipt["assignment_sha256"], digest)
        self.assertEqual(
            receipt["assignment_digest_serialization"],
            ASSIGNMENT_DIGEST_SERIALIZATION,
        )
        self.assertEqual(
            receipt["partition_proxy_counts"],
            {"subtrain": 4, "validation": 1, "test": 1},
        )

    def test_pairing_is_bidirectional(self):
        orphan = self.root / "test/test/steth_20210102_00_00_00_label.txt"
        orphan.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "missing_wavs"):
            build_hf_manifest(self.root, expected=self.expected)
        orphan.unlink()
        (self.root / f"{self.rows[0][0]}_label.txt").unlink()
        with self.assertRaisesRegex(RuntimeError, "missing_labels"):
            build_hf_manifest(self.root, expected=self.expected)

    def test_parser_and_date_proxy_boundaries(self):
        self.assertAlmostEqual(parse_hms("01:02:03.500"), 3_723.5)
        with self.assertRaises(ValueError):
            parse_hms("00:60:00")
        self.assertEqual(
            canonical_date_proxy("steth_20190809_11_04_31.wav"),
            "2019-08-09",
        )
        self.assertEqual(
            canonical_date_proxy("trunc_2020-01-02-00-00-00-L1_0.wav"),
            "2020-01-02",
        )
        bad = self.root / "bad_label.txt"
        bad.write_text("Normal 00:00:00 00:00:01\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unknown raw token"):
            parse_label_file(bad)
        bad.write_text("I 00:00:14 00:00:16\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "outside 15-second"):
            parse_label_file(bad)

    def test_header_audit(self):
        records, _ = build_hf_manifest(self.root, expected=self.expected)
        receipt = audit_wave_headers(self.root, records)
        self.assertEqual(receipt["headers_read"], 6)
        self.assertEqual(receipt["sample_rate"], 4_000)
        self.assertEqual(receipt["frames"], 60_000)

    def test_waveform_dependency_is_lazy(self):
        records, _ = build_hf_manifest(self.root, expected=self.expected)
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "torchaudio":
                raise ImportError("synthetic missing dependency")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded_import):
            with self.assertRaisesRegex(RuntimeError, "dependency-lazy"):
                load_hf_waveform(self.root, records[0])

    def test_fixed_p6_record_adapter(self):
        records, _ = build_hf_manifest(self.root, expected=self.expected)
        observed = next(
            record
            for record in records
            if record.recording_state is ObservationState.OBSERVED
        )
        empty = next(
            record
            for record in records
            if record.recording_state is ObservationState.EMPTY
        )
        ordered_records = (observed, empty)
        self.assertEqual(
            [record.recording_state for record in ordered_records],
            [ObservationState.OBSERVED, ObservationState.EMPTY],
        )
        time_map = torch.tensor(
            [
                [[0.0, 0.1], [0.1, 0.2]],
                [[0.0, 0.1], [0.1, 0.2]],
            ],
            dtype=torch.float32,
        )
        token_mask = torch.tensor([[True, True], [True, True]])
        supervision = records_to_p6_supervision(
            ordered_records, time_map, token_mask
        )
        self.assertEqual(tuple(supervision.targets.shape), (2, 2, 4))
        self.assertGreater(supervision.targets[0].sum().item(), 0)
        self.assertEqual(supervision.targets[1].sum().item(), 0)
        self.assertEqual(
            supervision.receipt["policy"], "paper_native_rasterized_ovr"
        )
        self.assertEqual(
            supervision.receipt["alignment"], "token_center_in_interval"
        )
        self.assertEqual(
            supervision.receipt["negative_semantics"],
            "source_task_constructed_not_raw_normal",
        )
        self.assertFalse(supervision.receipt["shared_label_eligible"])

    def test_record_schema_rejects_empty_as_observed_or_normal(self):
        with self.assertRaises(ValueError):
            HFSampleRecord(
                sample_id="hf:test:empty",
                source_split="test",
                partition="test",
                date_proxy="2020-01-01",
                group_id="date_proxy:2020-01-01",
                wav_relative_path="test/test/empty.wav",
                label_relative_path="test/test/empty_label.txt",
                raw_intervals=(),
                recording_state=ObservationState.OBSERVED,
            )
        with self.assertRaises(ValueError):
            HFSampleRecord(
                sample_id="hf:test:not-normal",
                source_split="test",
                partition="test",
                date_proxy="2020-01-01",
                group_id="date_proxy:2020-01-01",
                wav_relative_path="test/test/x.wav",
                label_relative_path="test/test/x_label.txt",
                raw_intervals=(HFRawInterval("I", 0.0, 0.1),),
                recording_state=ObservationState.EMPTY,
            )


if __name__ == "__main__":
    unittest.main()
