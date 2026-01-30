# -*- coding: utf-8 -*-
"""
RQ1.4: Feature fusion/task ablation over multiple tasks (aligned with RQ1 base)

- tasks: 10 task keys (ablation tasks)
- variant: only SB4_both (lambda_seed=0.1, lambda_boot=0.1)
- split_seed in {42,123,2025}, sample_rep in {1,2,3}

This runner assumes each task has its own packed dir:
  train_packed_data/<task>_LINEAR
"""
from datetime import datetime

from src.embedding.config.base_config import BaseConfig
from src.experiments.ablations.base_ablation_runner import RQ1Runner, RQ1RunConfig, Variant, TrainerParams


def main():
    config = BaseConfig.from_yaml()

    TASKS = [
        # "ab_baseline",
        # "ab_complete",
        "ab_widget_none",
        # "ab_widget_only_name",
        # "ab_widget_only_summary",
        # "ab_widget_name_summary",
        # "ab_activity_only_name",
        # "ab_activity_only_summary",
        # "ab_activity_simple_name",
        # "ab_activity_remove_suffix",
    ]

    # ✅ set your fixed best combo here
    BEST_ENCODER = "gin"
    BEST_REGULARIZER = "none"

    # ✅ fix a loss-weight learning policy here (edit if needed)
    LEARN_ALPHA = False
    LEARN_BETA = True
    LEARN_GAMMA = False
    INIT_A, INIT_B, INIT_G = 1.0, 0.5, 0.0

    only_variant = Variant(
        id="SB2_seed_only",
        encoder=BEST_ENCODER,
        regularizer=BEST_REGULARIZER,
        trainer=TrainerParams(
            lr=0.005, weight_decay=1e-5, epochs=80, batch_size=128, log_interval=10,
            learn_alpha=LEARN_ALPHA, learn_beta=LEARN_BETA, learn_gamma=LEARN_GAMMA,
            init_alpha=INIT_A, init_beta=INIT_B, init_gamma=INIT_G,
            lambda_seed=0.1, lambda_boot=0.0, boot_mix=0.5,
        ),
    )
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_cfg = RQ1RunConfig(
        rq_tag=f"rq14_feature_fusion_tasks_{run_ts}",
        tasks=TASKS,
        split_seeds=[42, 123, 2025],
        sample_reps=[17, 88, 241],
        variants=[only_variant],
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