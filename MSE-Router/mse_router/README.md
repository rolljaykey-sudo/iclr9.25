# Qwen–MOSEI conflict-aware router

The independent project root is `/gpfs/work/cpt/jiachenhou23/MSE-Router`.
The official Adapter repository is consumed only as an upstream dependency;
Router code and artifacts remain under this project root.

For a detailed Chinese description of the current architecture, design
evolution, and research narrative, see
[`ARCHITECTURE_STORY_ZH.md`](ARCHITECTURE_STORY_ZH.md).

This experiment is isolated from `MSE-Qwen-1.8B`, so an ongoing `cmcm`
baseline run cannot import partially changed scientific code.

The v2 implementation uses three independent four-token MSE projectors. A shared,
frozen Qwen-1.8B transformer produces seven-way ordinal logits for text, audio,
and vision. Calibrated class probabilities, three normalized JS divergences,
three normalized entropies, and three presence flags form the Router's 30 input
features. Router inputs are detached. In the final generative branch, the
original masked text-token sequence is retained as a semantic residual path;
the Router gates audio and vision pseudo tokens by
`number_of_present_modalities * weight`. The final loss remains language-model
cross entropy.

Training has three stages:

1. Train adapters, the ordinal head, and Router with generative CE plus 0.3
   times present-modality soft ordinal CE.
2. Fit three bounded scalar temperatures on a deterministic, video-group-safe
   10% calibration subset. An invalid or non-improving calibration aborts the
   dependent multi-seed run.
3. Freeze adapters, head, and temperatures; train only the Router with
   generative CE.

The entry point supports `preflight`, `train`, and `aggregate`, plus the
`full`, `no_conflict`, `no_uncertainty`, `predictions_only`,
`uncertainty_only`, and `uniform` routing variants:

```bash
.venv-repro/bin/python scripts/run_qwen_mosei_router.py --help
```

On XJTLU Slurm, the complete gated chain is submitted with:

```bash
cd /gpfs/work/cpt/jiachenhou23/MSE-Router
bash scripts/submit_qwen_mosei_router.sh
```

Artifacts are written only below `outputs/qwen-mosei-router-v2/`. Checkpoints
contain experiment parameters, temperatures, and stage metadata, never frozen
Qwen weights.
