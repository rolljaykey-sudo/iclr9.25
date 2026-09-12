#!/usr/bin/env python3
# MOSEI 实验配置入口：复用通用 ChatGLM3 流程，设置本版本的种子、训练参数或输出位置。
from dataclasses import dataclass
from pathlib import Path
import chatglm_setup as backbone

# 以基线入口为基础装配当前实验的配置、输出目录、种子和训练参数。
def configure():
    backbone.configure_harness()
    harness = backbone.harness
    harness.SEEDS = (1111, 1113, 1115)
    harness.DEFAULT_OUTPUT_ROOT = harness.PROJECT_DIR / "outputs"
    harness.ROUTER_SOURCE_FILES += (Path(__file__).resolve(), Path(__file__).resolve().parents[1] / "launch_spec.json")
    import mse_router.trainer as trainer
    # 覆盖基类中当前实验特有的训练参数，其余设置保持继承。
    @dataclass(frozen=True)
    class MoseiTrainingSettings(trainer.TrainingSettings):
        adapter_lr: float = 0.0001
        head_router_lr: float = 0.0001
    trainer.TrainingSettings = MoseiTrainingSettings
    return harness

if __name__ == "__main__":
    configure().main()
