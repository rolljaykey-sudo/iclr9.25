#!/usr/bin/env python3
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


def configure(target):
    h = baseline.configure()
    # Keep the original ChatGLM configuration provider, data loader, metrics,
    # prompts, trainer, and split. Only change the frozen backbone boundary.
    h.SEEDS = (1111, 1113, 1115)
    h.DEFAULT_MODEL = MODELS[target]
    h.DEFAULT_OUTPUT_ROOT = OUTPUT / target
    h.ROUTER_SOURCE_FILES += (Path(__file__).resolve(),)
    original_build = h.build_config

    def build_config(args, microbatch, accumulation):
        c = original_build(args, microbatch, accumulation)
        model_config = json.loads((args.model_path / "config.json").read_text())
        hidden = int(model_config["hidden_size"])
        c.router_hidden_size = hidden
        c.feature_dims = (hidden, *c.feature_dims[1:])
        c.router_backbone = "llama2" if target in ("llama2", "llama32") else "qwen18"
        return c

    h.build_config = build_config
    # The baseline loader has no missing-pad-token fallback (Llama has none).
    # Set a masked padding ID without changing truncation or sample contents.
    sys.path.insert(0, str(h.UPSTREAM_DIR))
    import data.load_data as loader_module
    tokenizer_factory = loader_module.AutoTokenizer

    class PaddingTokenizer:
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
        class Llama32Router(BackboneMseRouter):
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
