import unittest
from pathlib import Path

from baseline.pafa.table2_benchmark_controls import (
    BenchmarkConfig,
    _learning_rate,
    benchmark_split_reference,
    default_config,
    load_benchmark_training_samples,
)


ROOT = Path(__file__).resolve().parents[1]


class Table2BenchmarkControlsTest(unittest.TestCase):
    def test_learning_rate_matches_historical_jh2_epochs(self) -> None:
        for epoch, expected in (
            (1, 4.9950717542496084e-5),
            (25, 2.5025e-5),
            (50, 5e-8),
        ):
            self.assertAlmostEqual(_learning_rate(epoch), expected, places=16)

    def test_active_sources_and_selection_dataset(self) -> None:
        icbhi = default_config(ROOT, "icbhi_only")
        spr = default_config(ROOT, "sprsound_only")
        coarse = default_config(ROOT, "coarse_spr")
        native = default_config(ROOT, "native_attributes")
        self.assertEqual(icbhi.active_datasets, ("icbhi",))
        self.assertEqual(spr.active_datasets, ("sprsound",))
        self.assertEqual(coarse.active_datasets, ("icbhi", "sprsound"))
        self.assertEqual(native.active_datasets, ("icbhi", "sprsound"))
        self.assertEqual(icbhi.selection_dataset, "icbhi")
        self.assertEqual(spr.selection_dataset, "sprsound")

    def test_original_jh2_seed42_sample_partitions(self) -> None:
        for variant, expected in (
            ("icbhi_only", {"icbhi": (3174, 968)}),
            ("sprsound_only", {"sprsound": (5219, 1437)}),
            ("coarse_spr", {"icbhi": (3174, 968), "sprsound": (5219, 1437)}),
            ("native_attributes", {"icbhi": (3174, 968), "sprsound": (5219, 1437)}),
        ):
            config = default_config(ROOT, variant)
            samples = load_benchmark_training_samples(config)
            reference = benchmark_split_reference(config, samples)
            for dataset, (subtrain, validation) in expected.items():
                support = reference["datasets"][dataset]["support"]
                self.assertEqual(support["subtrain"]["units"], subtrain)
                self.assertEqual(support["validation"]["units"], validation)
                self.assertFalse(
                    set(reference["datasets"][dataset]["groups"]["subtrain"])
                    & set(reference["datasets"][dataset]["groups"]["validation"])
                )

    def test_multiseed_config_uses_matching_reference_and_output(self) -> None:
        for seed in (0, 1):
            with self.subTest(seed=seed):
                config = default_config(ROOT, "native_attributes", seed)
                config.validate()
                self.assertEqual(config.seed, seed)
                self.assertEqual(config.core.model_seed, seed)
                self.assertEqual(config.core.base_config().seed, seed)
                self.assertEqual(
                    config.full_reference,
                    ROOT / "result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed" / f"seed_{seed}",
                )
                self.assertEqual(config.to_dict()["full_reference"], str(config.full_reference))
                self.assertEqual(
                    config.output_dir,
                    ROOT / "result/reproduce/pafa_joint_hierarchy/PAFA_BENCHMARK_4COND_multiseed/native_attributes" / f"seed_{seed}",
                )

    def test_original_jh2_multiseed_sample_partitions(self) -> None:
        for seed, expected_icbhi in ((0, (3636, 506)), (1, (2880, 1262))):
            with self.subTest(seed=seed):
                config = default_config(ROOT, "coarse_spr", seed)
                samples = load_benchmark_training_samples(config)
                reference = benchmark_split_reference(config, samples)
                for dataset, expected in (("icbhi", expected_icbhi), ("sprsound", (5219, 1437))):
                    support = reference["datasets"][dataset]["support"]
                    self.assertEqual(
                        (support["subtrain"]["units"], support["validation"]["units"]), expected
                    )
                    groups = reference["datasets"][dataset]["groups"]
                    self.assertFalse(set(groups["subtrain"]) & set(groups["validation"]))

    def test_default_seed_and_local_recipe_preserved(self) -> None:
        config = BenchmarkConfig(
            repo_root=ROOT,
            variant="native_attributes",
            output_dir=ROOT / "result/reproduce/pafa_joint_hierarchy/PAFA_BENCHMARK_4COND_seed42/native_attributes/seed_42",
        )
        config.validate()
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.device, "mps")
        self.assertEqual(config.cpu_threads, 4)
        self.assertEqual(config.core.batch_size, 32)


if __name__ == "__main__":
    unittest.main()
