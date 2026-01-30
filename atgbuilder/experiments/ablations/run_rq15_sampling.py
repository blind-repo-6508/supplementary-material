# -*- coding: utf-8 -*-
"""
RQ1.5: Sampling Ablation
S0: no sampling (baseline)
S1: random 1:1
S2: hard negatives 1:1 (same-source)
"""
from datetime import datetime

from src.embedding.config.base_config import BaseConfig
from src.experiments.ablations.base_ablation_runner import (
    RQ1Runner,
    RQ1RunConfig,
    Variant,
    TrainerParams,
)


def main():
    config = BaseConfig.from_yaml()

    variants = [
        Variant(id="S0_nosample", encoder="gin", regularizer="none", sampling="none"),
        # Variant(id="S1_random_1to1", encoder="gin", regularizer="none", sampling="random_1to1"),
        Variant(id="S2_hardsrc_1to1", encoder="gin", regularizer="none", sampling="hard_same_source_1to1"),
    ]

    tp = TrainerParams(
        lr=0.005,
        weight_decay=1e-5,
        epochs=80,
        batch_size=128,
        log_interval=10,
        learn_alpha=False,
        learn_beta=True,
        learn_gamma=False,
        init_alpha=1.0,
        init_beta=0.5,
        init_gamma=0.0,
        lambda_seed=0.1,
        lambda_boot=0.0,
        boot_mix=0.5,
    )
    for v in variants:
        v.trainer = tp

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_cfg = RQ1RunConfig(
        rq_tag=f"rq15_sampling_{run_ts}",
        tasks=["ab_activity_only_summary"],
        split_seeds=[42, 123, 2025],
        sample_reps=[17, 88, 241],
        variants=variants,
        pack_dir_name="train_packed_data",
        method_suffix="LINEAR",
        split_ratios=(0.7, 0.1, 0.2),
        max_neg_when_pos0=0,
        threshold_grid=199,
    )

    runner = RQ1Runner(config, run_cfg)
    runner.run()


if __name__ == "__main__":
    main()
