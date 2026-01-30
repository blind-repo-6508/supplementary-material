# -*- coding: utf-8 -*-
"""
RQ1.3: Seed-consistency / Bootstrapping strategy ablation (aligned with RQ1 base)

Variants:
SB1_base      (0, 0)
SB2_seed_only (0.1, 0)
SB3_boot_only (0, 0.1)
SB4_both      (0.1, 0.1)

- task: ab_complete
- encoder/regularizer fixed to your best setting
- split_seed in {42,123,2025}, sample_rep in {1,2,3}
"""
from datetime import datetime

from src.embedding.config.base_config import BaseConfig
from src.experiments.ablations.base_ablation_runner import RQ1Runner, RQ1RunConfig, Variant, TrainerParams


def main():
    config = BaseConfig.from_yaml()

    # ✅ set your best combo here
    BEST_ENCODER = "gin"
    BEST_REGULARIZER = "none"

    # ✅ fix a loss-weight learning policy for this RQ (edit if needed)
    LEARN_ALPHA = False
    LEARN_BETA = True
    LEARN_GAMMA = False
    INIT_A, INIT_B, INIT_G = 1.0, 0.5, 0.0

    variants = [
        # Variant(
        #     id="SB1_base",
        #     encoder=BEST_ENCODER,
        #     regularizer=BEST_REGULARIZER,
        #     trainer=TrainerParams(
        #         lr=0.005, weight_decay=1e-5, epochs=100, batch_size=128, log_interval=10,
        #         learn_alpha=LEARN_ALPHA, learn_beta=LEARN_BETA, learn_gamma=LEARN_GAMMA,
        #         init_alpha=INIT_A, init_beta=INIT_B, init_gamma=INIT_G,
        #         lambda_seed=0.0, lambda_boot=0.0, boot_mix=0.5,
        #     ),
        # ),
        # Variant(
        #     id="SB2_seed_only",
        #     encoder=BEST_ENCODER,
        #     regularizer=BEST_REGULARIZER,
        #     trainer=TrainerParams(
        #         lr=0.005, weight_decay=1e-5, epochs=80, batch_size=128, log_interval=10,
        #         learn_alpha=LEARN_ALPHA, learn_beta=LEARN_BETA, learn_gamma=LEARN_GAMMA,
        #         init_alpha=INIT_A, init_beta=INIT_B, init_gamma=INIT_G,
        #         lambda_seed=0.1, lambda_boot=0.0, boot_mix=0.5,
        #     ),
        # ),
        # Variant(
        #     id="SB3_boot_only",
        #     encoder=BEST_ENCODER,
        #     regularizer=BEST_REGULARIZER,
        #     trainer=TrainerParams(
        #         lr=0.005, weight_decay=1e-5, epochs=80, batch_size=128, log_interval=10,
        #         learn_alpha=LEARN_ALPHA, learn_beta=LEARN_BETA, learn_gamma=LEARN_GAMMA,
        #         init_alpha=INIT_A, init_beta=INIT_B, init_gamma=INIT_G,
        #         lambda_seed=0.0, lambda_boot=0.1, boot_mix=0.5,
        #     ),
        # ),
        Variant(
            id="SB4_both",
            encoder=BEST_ENCODER,
            regularizer=BEST_REGULARIZER,
            trainer=TrainerParams(
                lr=0.005, weight_decay=1e-5, epochs=80, batch_size=128, log_interval=10,
                learn_alpha=LEARN_ALPHA, learn_beta=LEARN_BETA, learn_gamma=LEARN_GAMMA,
                init_alpha=INIT_A, init_beta=INIT_B, init_gamma=INIT_G,
                lambda_seed=0.1, lambda_boot=0.1, boot_mix=0.5,
            ),
        ),
    ]
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_cfg = RQ1RunConfig(
        rq_tag=f"rq13_seed_boot_strategy_{run_ts}",
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
