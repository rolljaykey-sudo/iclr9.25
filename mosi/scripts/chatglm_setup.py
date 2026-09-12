"""为当前数据集装配固定的 ChatGLM3 模型接口和文件校验。"""
from pathlib import Path
import experiment as harness

PROJECT_DIR = Path(__file__).resolve().parents[1]


# 由数据集入口先配置骨干，再覆盖该数据集特有的参数。
def configure_harness():
    upstream = harness.ADAPTER_DIR / "MSE-ChatGLM3-6B"
    harness.UPSTREAM_DIR = upstream
    harness.DEFAULT_MODEL = harness.DEFAULT_MODEL
    harness.DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "outputs"
    # 预检绑定本地源码及上游实际使用的配置、数据和 ChatGLM 实现。
    harness.ROUTER_SOURCE_FILES = (
        *(PROJECT_DIR / "mse_router" / name for name in
          ("math_utils.py", "sequence.py", "data.py", "model.py", "trainer.py")),
        PROJECT_DIR / "scripts" / "experiment.py",
        Path(__file__).resolve(),
        upstream / "config" / "config_regression.py",
        upstream / "data" / "load_data.py",
        upstream / "utils" / "metricsTop.py",
        *(upstream / "models" / "ChatGLM3" / name for name in
          ("configuration_chatglm.py", "modeling_chatglm.py", "tokenization_chatglm.py")),
    )
    original_build_config = harness.build_config

    # 统一冻结骨干的隐藏维度与上游实现位置。
    def build_config(args, microbatch, accumulation):
        config = original_build_config(args, microbatch, accumulation)
        config.router_hidden_size = 4096
        config.router_backbone = "chatglm3"
        config.router_upstream_dir = str(upstream.resolve())
        return config

    # 验证数据指纹及 ChatGLM3 权重分片；元数据将用于训练前的一致性检查。
    def validate_files(args, hash_dataset):
        dataset, model = args.dataset_path.resolve(), args.model_path.resolve()
        if not dataset.is_file() or dataset.stat().st_size != harness.EXPECTED_DATASET_SIZE:
            raise RuntimeError(f"unexpected or missing dataset: {dataset}")
        digest = harness.sha256_file(dataset) if hash_dataset else None
        if digest is not None and digest != harness.EXPECTED_DATASET_SHA256:
            raise RuntimeError(f"dataset SHA-256 mismatch: {digest}")
        required = (
            "config.json", "pytorch_model.bin.index.json",
            *(f"pytorch_model-{part:05d}-of-00007.bin" for part in range(1, 8)),
            "tokenizer.model", "tokenizer_config.json",
        )
        missing = [name for name in required if not (model / name).is_file()]
        if missing:
            raise RuntimeError(f"missing ChatGLM3 files: {missing}")
        return {
            "dataset": {"path": str(dataset), "size": dataset.stat().st_size,
                        "mtime_ns": dataset.stat().st_mtime_ns, "sha256": digest},
            "model": {"backbone": "chatglm3", "path": str(model), "files": {
                name: {"size": (model/name).stat().st_size, "mtime_ns": (model/name).stat().st_mtime_ns}
                for name in required}},
        }

    harness.build_config = build_config
    harness.validate_files = validate_files
