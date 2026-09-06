# Third-party source

`MSE-Adapter/` contains the configuration, processed-data loader, metrics,
utility functions and ChatGLM3 implementation actually used by the original
Router experiments, copied from the local snapshot of
[AZYoung233/MSE-Adapter](https://github.com/AZYoung233/MSE-Adapter).
Each included backbone directory retains its original MIT `LICENSE`.
File checksums are recorded in `SOURCE_MANIFEST.json`.

The Router uses external Qwen-1.8B, Llama-2-7b-hf and ChatGLM3-6B-base
checkpoints. Their weights, tokenizers and processed datasets are supplied
separately under their respective distribution terms.
