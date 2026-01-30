# -*- coding: utf-8 -*-
"""
RQ1.2: Loss weight learning strategy (aligned with RQ1 base)

Goal:
- Study loss-weight learning strategy under the *best backbone* found in RQ1.1.
- Since we disable contrastive regularization here, we set gamma=0 and do NOT learn gamma.

Variants:
LW1: alpha fixed (=1), learn beta (attr weight)
LW2: learn alpha (exist weight) and beta (attr weight)
"""
from datetime import datetime

from src.embedding.config.base_config import BaseConfig
from src.experiments.ablations.base_ablation_runner import RQ1Runner, RQ1RunConfig, Variant, TrainerParams


def main():
    config = BaseConfig.from_yaml()

    # ✅ Best backbone from RQ1.1
    BEST_ENCODER = "gin"  # "gcn" or "gin"
    BEST_REGULARIZER = "none"  # "none" / "moco" / "fixed_moco"  (use "none" for all later ablations)

    variants = [
        # Variant(
        #     id="LW1_fixA_learnB",
        #     encoder=BEST_ENCODER,
        #     regularizer=BEST_REGULARIZER,
        #     trainer=TrainerParams(
        #         lr=0.005, weight_decay=1e-5, epochs=100, batch_size=128, log_interval=10,
        #         learn_alpha=False, learn_beta=True, learn_gamma=False,
        #         init_alpha=1.0, init_beta=0.5, init_gamma=0.0,
        #         lambda_seed=0.0, lambda_boot=0.0, boot_mix=0.5,
        #     ),
        # ),
        # Variant(
        #     id="LW2_learnA_learnB",
        #     encoder=BEST_ENCODER,
        #     regularizer=BEST_REGULARIZER,
        #     trainer=TrainerParams(
        #         lr=0.005, weight_decay=1e-5, epochs=80, batch_size=128, log_interval=10,
        #         learn_alpha=True, learn_beta=True, learn_gamma=False,
        #         init_alpha=1.0, init_beta=0.5, init_gamma=0.0,
        #         lambda_seed=0.0, lambda_boot=0.0, boot_mix=0.5,
        #     ),
        # ),
        Variant(
            id="LW3_fixA_B_0",
            encoder=BEST_ENCODER,
            regularizer=BEST_REGULARIZER,
            trainer=TrainerParams(
                lr=0.005, weight_decay=1e-5, epochs=80, batch_size=128, log_interval=10,
                learn_alpha=False, learn_beta=False, learn_gamma=False,
                init_alpha=1.0, init_beta=0.0, init_gamma=0.0,
                lambda_seed=0.0, lambda_boot=0.0, boot_mix=0.5,
            ),
        ),
    ]

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_cfg = RQ1RunConfig(
        rq_tag=f"rq12_loss_weight_strategy_{run_ts}",
        tasks=["ab_complete"],
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
