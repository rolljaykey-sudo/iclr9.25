#!/usr/bin/env python3
# MOSI 实验配置入口：复用通用 ChatGLM3 流程，设置本版本的种子、训练参数或输出位置。
"""Additional seeds for the unchanged MOSI lr1e-3 test-best full40 protocol."""
from pathlib import Path
from run_chatglm3_mosi import configure

if __name__ == "__main__":
    harness = configure()
    harness.SEEDS = (1115,)
    harness.ROUTER_SOURCE_FILES += (Path(__file__).resolve(),)
    harness.main()
