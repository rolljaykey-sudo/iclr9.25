#!/usr/bin/env python3
# SIMS 骨干迁移入口：沿用基线实验配置，适配目标模型的隐藏维度与分词器。
"""Transfer the verified ChatGLM SIMS configuration to another frozen backbone."""
import argparse
import json
import sys
from pathlib import Path

import run_chatglm3_sims as baseline

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT.parents[1] / "outputs" / "sims_backbone_lr1e3_d00_sat3090_20260912"
MODELS = {
    "llama2": Path("/gpfs/work/cpt/jiachenhou23/models/Meta/Llama-2-7b-hf"),
    "llama32": Path("/gpfs/work/cpt/jiachenhou23/models/Meta/Llama-3.2-3B"),
    "qwen18": Path("/gpfs/work/cpt/jiachenhou23/models/Qwen/Qwen-1_8B"),
}


# 以基线入口为基础装配当前实验的配置、输出目录、种子和训练参数。
def configure(target):
    h = baseline.configure()
    # 沿用 ChatGLM 基线的配置来源、数据加载器、指标、
    # 提示词、训练器与划分，只适配冻结骨干的接口。
    h.SEEDS = (1111, 1113, 1115)
    h.DEFAULT_MODEL = MODELS[target]
    h.DEFAULT_OUTPUT_ROOT = OUTPUT / target
    h.ROUTER_SOURCE_FILES += (Path(__file__).resolve(),)
    original_build = h.build_config

    # 在上游配置基础上设置当前实验的数据、骨干维度、批量与路由参数。
    def build_config(args, microbatch, accumulation):
        c = original_build(args, microbatch, accumulation)
        model_config = json.loads((args.model_path / "config.json").read_text())
        hidden = int(model_config["hidden_size"])
        c.router_hidden_size = hidden
        c.feature_dims = (hidden, *c.feature_dims[1:])
        c.router_backbone = "llama2" if target in ("llama2", "llama32") else "qwen18"
        return c

    h.build_config = build_config
    # 基线加载器没有为缺少 pad token 的分词器提供回退；
    # 这里补充可被掩码排除的填充 ID，保持截断方式和样本内容不变。
    sys.path.insert(0, str(h.UPSTREAM_DIR))
    import data.load_data as loader_module
    tokenizer_factory = loader_module.AutoTokenizer

    # 封装分词器加载，为缺少 pad token 的模型补上可被注意力掩码排除的填充 ID。
    class PaddingTokenizer:
        # 加载分词器后检查 pad token；缺失时复用结束 token 作为填充标记。
        @staticmethod
        def from_pretrained(*args, **kwargs):
            tokenizer = tokenizer_factory.from_pretrained(*args, **kwargs)
            if tokenizer.pad_token_id is None:
                eos = tokenizer.eos_token_id
                if eos is None:
                    eos = tokenizer.convert_tokens_to_ids("<|endoftext|>")
                tokenizer.pad_token_id = int(eos)
            return tokenizer

    loader_module.AutoTokenizer = PaddingTokenizer
    import mse_router.model as model_module
    from mse_router.backbone_model import BackboneMseRouter, QwenMseRouter

    if target == "qwen18":
        model_module.QwenMseRouter = QwenMseRouter
    elif target == "llama32":
        # 为 Llama3.2 覆盖骨干加载接口，其余路由计算沿用统一实现。
        class Llama32Router(BackboneMseRouter):
            # 按当前实验入口加载对应骨干和分词器，并适配其特殊 token 与输入接口。
            def _load_backbone(self, args):
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                tokenizer = AutoTokenizer.from_pretrained(
                    args.pretrain_LM, padding_side="left", local_files_only=True,
                    trust_remote_code=False)
                model = AutoModelForCausalLM.from_pretrained(
                    args.pretrain_LM, local_files_only=True, trust_remote_code=False,
                    torch_dtype=torch.float16, attn_implementation="eager").half()
                return tokenizer, model
        model_module.QwenMseRouter = Llama32Router
    else:
        model_module.QwenMseRouter = BackboneMseRouter

    # 检查数据文件大小、可选 SHA-256 及骨干文件完整性，并记录文件元数据。
    def validate_files(args, hash_dataset):
        dataset = args.dataset_path.resolve()
        if not dataset.is_file() or dataset.stat().st_size != h.EXPECTED_DATASET_SIZE:
            raise RuntimeError(f"Unexpected dataset: {dataset}")
        digest = h.sha256_file(dataset) if hash_dataset else None
        if digest is not None and digest != h.EXPECTED_DATASET_SHA256:
            raise RuntimeError("Dataset SHA256 differs from verified SIMS baseline")
        model = args.model_path.resolve()
        required = ["config.json", "tokenizer_config.json"]
        index = model / "model.safetensors.index.json"
        if index.exists():
            required += [index.name, *sorted(set(json.loads(index.read_text())["weight_map"].values()))]
        else:
            required += ["model.safetensors"]
        required += ["qwen.tiktoken"] if target == "qwen18" else ["tokenizer.json"]
        missing = [f for f in required if not (model / f).is_file()]
        if missing:
            raise RuntimeError(f"Missing {target} files: {missing}")
        return {
            "dataset": {"path": str(dataset), "size": dataset.stat().st_size,
                        "mtime_ns": dataset.stat().st_mtime_ns, "sha256": digest},
            "model": {"backbone": target, "path": str(model), "files": {
                f: {"size": (model / f).stat().st_size,
                    "mtime_ns": (model / f).stat().st_mtime_ns} for f in required}},
        }

    h.validate_files = validate_files
    return h


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--target", choices=MODELS, required=True)
    target_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    configure(target_args.target).main()
