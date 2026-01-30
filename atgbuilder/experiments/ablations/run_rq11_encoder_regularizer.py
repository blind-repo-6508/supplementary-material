# -*- coding: utf-8 -*-
"""
RQ1.1: Encoder & Representation Regularizer Ablation (aligned with RQ1 base)

Variants:
E1: GCN + None
E2: GCN + MoCo
E3: GCN + Fixed-MoCo
E4: GIN + None
E5: GIN + MoCo
E6: GIN + Fixed-MoCo

- task: ab_complete
- split_seed in {42,123,2025}
- sample_rep in {17, 88, 241}
- per-graph 1:1 sampling for train/val/test
- post-hoc t* selected on VAL; report TEST@t* and TEST@0.5
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
        Variant(id="E1", encoder="gcn", regularizer="none"),
        Variant(id="E2", encoder="gcn", regularizer="moco"),
        Variant(id="E3", encoder="gcn", regularizer="fixed_moco"),
        Variant(id="E4", encoder="gin", regularizer="none"),
        Variant(id="E5", encoder="gin", regularizer="moco"),
        Variant(id="E6", encoder="gin", regularizer="fixed_moco"),
    ]

    # ========= Common trainer params for RQ1.1 =========
    # moco / fixed_moco: learn beta & gamma; gamma init > 0 (log-param)
    tp_with_reg = TrainerParams(
        lr=0.005,
        weight_decay=1e-5,
        epochs=80,
        batch_size=128,
        log_interval=10,
        learn_alpha=False,
        learn_beta=True,
        learn_gamma=True,  # ✅ learn gamma only when using MoCo-family regularizer
        init_alpha=1.0,
        init_beta=0.5,
        init_gamma=0.1,  # ✅ must be > 0 when learn_gamma=True
        lambda_seed=0.0,
        lambda_boot=0.0,
        boot_mix=0.5,
    )

    # regularizer="none": DO NOT learn gamma; fix gamma=0
    tp_none = TrainerParams(
        lr=0.005,
        weight_decay=1e-5,
        epochs=80,
        batch_size=128,
        log_interval=10,
        learn_alpha=False,
        learn_beta=True,
        learn_gamma=False,  # ✅关键：不学 gamma
        init_alpha=1.0,
        init_beta=0.5,
        init_gamma=0.0,  # ✅关键：固定为 0，不走 log-param，不会 clamp 到 exp(-3)
        lambda_seed=0.0,
        lambda_boot=0.0,
        boot_mix=0.5,
    )

    for v in variants:
        v.trainer = tp_none if v.regularizer == "none" else tp_with_reg

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")  # 例如 20251221_213607

    run_cfg = RQ1RunConfig(
        rq_tag=f"rq11_encoder_regularizer_{run_ts}",
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
