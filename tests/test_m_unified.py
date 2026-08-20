import unittest

import torch

from baseline.four_dataset_frozen_encoder.data import Sample
from baseline.multidataset_pipeline.m_unified import (
    eligible_node_loss,
    map_hf_window,
    map_native_sample,
    regroup_hf_native_probabilities,
    regroup_native_probabilities,
)


def sample(dataset: str, targets: dict, metadata: dict) -> Sample:
    return Sample(
        sample_id=f"{dataset}:example",
        dataset=dataset,
        partition="subtrain",
        group_id="group",
        audio_path="unused.wav",
        crop_start_s=None,
        crop_end_s=None,
        targets=targets,
        metadata=metadata,
    )


class MUnifiedMappingTest(unittest.TestCase):
    def test_native_and_hf_mapping_boundaries(self):
        icbhi = map_native_sample(sample("icbhi", {"icbhi_flat4": 3}, {}))
        self.assertEqual(
            (icbhi["level1_target"], icbhi["crackle_target"], icbhi["wheeze_target"]),
            (1, 1, 1),
        )

        rhonchi = map_native_sample(
            sample("sprsound", {"spr_seven": 1}, {"raw_label": "Rhonchi"})
        )
        self.assertTrue(rhonchi["level1_eligible"])
        self.assertTrue(rhonchi["other_eligible"])
        self.assertFalse(rhonchi["crackle_eligible"])
        self.assertFalse(rhonchi["wheeze_eligible"])

        unresolved = map_native_sample(
            sample("kauh", {"kauh_raw9": 6}, {"raw_sound": "Crep"})
        )
        self.assertFalse(any(unresolved[f"{node}_eligible"] for node in ("level1", "crackle", "wheeze", "other")))

        hf_sample = sample(
            "hf_lung",
            {},
            {
                "raw_intervals": [
                    ["D", 0.0, 2.0],
                    ["Wheeze", 1.0, 3.0],
                    ["Rhonchi", 0.5, 1.5],
                ]
            },
        )
        positive = map_hf_window(hf_sample, 0, 0.0, 2.0)
        self.assertFalse(positive["level1_eligible"])
        self.assertEqual(
            (positive["crackle_target"], positive["wheeze_target"], positive["other_target"]),
            (1, 1, 1),
        )
        gap = map_hf_window(hf_sample, 4, 4.0, 6.0)
        self.assertFalse(any(gap[f"{node}_eligible"] for node in ("level1", "crackle", "wheeze", "other")))

    def test_equal_active_node_loss_and_masking(self):
        logits = {
            "level1": torch.tensor([[[0.0, 1.0], [2.0, 0.0]]], requires_grad=True),
            "crackle": torch.tensor([[0.2, -0.4]], requires_grad=True),
            "wheeze": torch.tensor([[0.1, 0.3]], requires_grad=True),
            "other": torch.tensor([[0.5, -0.2]], requires_grad=True),
        }
        targets = torch.zeros(1, 2, 4)
        targets[0, 0, 0] = 1
        targets[0, 1, 1] = 1
        eligible = torch.zeros(1, 2, 4, dtype=torch.bool)
        eligible[0, 0, 0] = True
        eligible[0, 1, 1] = True
        loss, named = eligible_node_loss(logits, targets, eligible)
        self.assertEqual(set(named), {"level1", "crackle"})
        loss.backward()
        self.assertIsNotNone(logits["level1"].grad)
        self.assertIsNotNone(logits["crackle"].grad)
        self.assertIsNone(logits["wheeze"].grad)

    def test_posthoc_regrouping_does_not_split_hf_cas(self):
        icbhi = regroup_native_probabilities(
            "icbhi", {"normal": 0.2, "crackle": 0.3, "wheeze": 0.1, "both": 0.4}
        )
        self.assertAlmostEqual(icbhi["crackle"], 0.7)
        hf = regroup_hf_native_probabilities({"CAS": 0.8, "DAS": 0.25})
        self.assertEqual(hf["crackle"], 0.25)
        self.assertIsNone(hf["wheeze"])
        self.assertIsNone(hf["other"])


if __name__ == "__main__":
    unittest.main()
