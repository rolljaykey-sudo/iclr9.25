"""Lightweight configuration and local tokenizer checks; never loads weights."""
import ast
import json
import sys
from dataclasses import asdict
from pathlib import Path

import run_transfer_sims as transfer

target = sys.argv[1]
sys.argv = [sys.argv[0], "preflight"]
harness = transfer.configure(target)
args = harness.parse_args()
config = harness.build_config(args, 4, 4)
files = harness.validate_files(args, hash_dataset=False)
import mse_router.trainer as trainer
import mse_router.model as model_module
import data.load_data as loader_module
import transformers

settings = asdict(trainer.TrainingSettings())
assert settings == transfer.baseline.SPEC["training_settings"], settings
assert config.datasetName == "sims" and config.language == "cn"
assert config.feature_dims[1:] == (33, 709)
assert config.seq_lens == (50, 400, 55)
assert config.task_specific_prompt == transfer.baseline.SPEC["task_specific_prompt"]
assert config.diagnostic_prompt == transfer.baseline.SPEC["diagnostic_prompt"]
assert harness.SEEDS == (1111, 1113, 1115)
assert config.a_lstm_dropout == config.v_lstm_dropout == 0.0
assert not (args.output_root / "task_1111").exists()
tokenizer = loader_module.AutoTokenizer.from_pretrained(
    str(args.model_path), local_files_only=True, trust_remote_code=True)
assert tokenizer.pad_token_id is not None
encoded = tokenizer(config.task_specific_prompt, add_special_tokens=False)
assert encoded["input_ids"]
if target == "llama32":
    mc = transformers.AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    assert mc.hidden_size == config.router_hidden_size == 3072
for path in transfer.ROOT.rglob("*.py"):
    ast.parse(path.read_text(), filename=str(path))
report = {
    "status": "ok", "target": target, "python": sys.executable,
    "transformers": transformers.__version__, "settings": settings,
    "router_class": model_module.QwenMseRouter.__name__,
    "feature_dims": config.feature_dims, "seq_lens": config.seq_lens,
    "tokenizer_class": type(tokenizer).__name__, "pad_token_id": tokenizer.pad_token_id,
    "files": files,
}
(transfer.ROOT / f"{target}_cpu_validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "files"}))
