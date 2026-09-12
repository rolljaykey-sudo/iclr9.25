#!/usr/bin/env python3
# MOSI 实验配置入口：复用通用 ChatGLM3 流程，设置本版本的种子、训练参数或输出位置。
"""ChatGLM3 + MOSI, Wasserstein V4 joint training, both learning rates 1e-4."""
from dataclasses import dataclass
from pathlib import Path
import run_backbone_mosei_router as backbone


# 以基线入口为基础装配当前实验的配置、输出目录、种子和训练参数。
def configure():
    backbone.configure_harness("chatglm3")
    harness = backbone.harness
    harness.DEFAULT_DATASET = Path("/gpfs/work/cpt/jiachenhou23/datasets/MSA/data/CMU-MOSI/Processed/unaligned_50.pkl")
    harness.DEFAULT_OUTPUT_ROOT = Path('/gpfs/work/cpt/jiachenhou23/mse router new conflict/outputs/chatglm3-mosi-lr1e4-testf1max-full40-20260910')
    harness.EXPECTED_SPLITS = {"train": 1284, "valid": 229, "test": 686}
    harness.EXPECTED_DATASET_SIZE = 554181170
    harness.EXPECTED_DATASET_SHA256 = "78e0f8b5ef8ff71558e7307848fc1fa929ecb078203f565ab22b9daab2e02524"
    harness.SEEDS = (1111, 1113, 1115)
    harness.ROUTER_SOURCE_FILES += (Path(__file__).resolve(), Path(harness.__file__).resolve(), Path(__file__).resolve().parents[1] / "launch_spec.json")
    original_build_config = harness.build_config

    # 在上游配置基础上设置当前实验的数据、骨干维度、批量与路由参数。
    def build_config(args, microbatch, accumulation):
        config = original_build_config(args, microbatch, accumulation)
        config.datasetName = "mosi"
        config.seq_lens = (50, 375, 500)
        config.feature_dims = (4096, 5, 20)
        config.train_samples = 1284
        config.language = "en"
        return config

    harness.build_config = build_config
    import mse_router.trainer as trainer

    # 覆盖基类中当前实验特有的训练参数，其余设置保持继承。
    @dataclass(frozen=True)
    class MosiTrainingSettings(trainer.TrainingSettings):
        adapter_lr: float = 1e-4
        head_router_lr: float = 1e-4

    trainer.TrainingSettings = MosiTrainingSettings
    return harness


if __name__ == "__main__":
    configure().main()
