# Archive validation — 2026-09-12

This upload packages existing experiment sources and adds documentation. It does
not launch training or change the model, loss, configuration, or training code.

## Source identity and syntax

`python tools/verify_snapshot.py` passed:

```json
{"status": "ok", "hashed_files": 42, "python": 31, "json": 8, "slurm": 3}
```

The archive contains exact copies of the original router, scripts, configuration,
recorded revision metadata and required upstream files. All 17 source hashes
recorded by each of the eight available training manifests match the corresponding
archived files. Their environment and split records are in
`provenance/source_snapshot.json`; the ninth job had not produced a training
manifest at export time.

The new verifier uses Python's standard library for SHA-256, AST parsing and JSON
parsing, and `bash -n` for Slurm scripts. No training code is executed by it.

The existing repository integrity check, `python scripts/check_source.py` from
the repository root, also passed: **88 original files verified and 63 Python files
parsed**. The earlier V2 source archive is preserved.

The staged whitespace check reports existing trailing whitespace in bundled
upstream files. Those bytes are retained to preserve source hashes. The check
passes when restricted to all other added/modified files.

## CPU configuration and tokenizer checks

For each target, a separate process used its original project environment,
pointed `MSE_ADAPTER_ROOT` to the archived `upstream_adapter/`, imported the
archived entry point and built the SIMS runtime configuration. Checks covered:

- archived trainer and upstream loader import locations;
- all effective `TrainingSettings` matching the original JSON;
- SIMS dataset identity, full Router, zero LSTM dropout and seeds 1111/1113/1115;
- sequence lengths 50/400/55 and audio/vision dimensions 33/709;
- target-specific hidden dimensions and local tokenizer loading;
- non-null padding token ID and successful Chinese prompt tokenization.

| Target | Hidden size | Tokenizer | Result |
| --- | ---: | --- | --- |
| qwen18 | 2048 | QWenTokenizer | passed |
| llama2 | 4096 | LlamaTokenizerFast | passed |
| llama32 | 3072 | PreTrainedTokenizerFast | passed |

These checks loaded neither dataset contents nor model weights. Llama 3.2's
environment emitted a torchvision image-extension symbol warning while importing
dependencies; its configuration and tokenizer checks still completed successfully.
The experiment consumes processed feature tensors rather than torchvision image
decoding, but this check does not establish that the torchvision image extension
works.

Dependency pins were read from the original environments, not inferred from the
older parent requirements files. They are not complete transitive lockfiles.
No clean-environment installation or fresh GPU training was performed for this
documentation/archive upload. Existing training manifests provide historical
execution evidence, not a new end-to-end validation of a relocated checkout.
