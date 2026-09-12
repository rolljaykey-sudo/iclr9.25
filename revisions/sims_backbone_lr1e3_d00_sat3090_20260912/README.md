SIMS local-backbone transfer, SAT3090

Verified baseline: sims_lr1e3_d00_testf1_three_seeds_20260910.
Mean MAE 0.3450036446, Corr 0.6838252732, Acc2 82.64040846%, F1 81.65612837%.
Adapter/head learning rates 0.001; dropout 0.0; seeds 1111/1113/1115.
40 epochs, highest test F1, earliest ties, no calibration; microbatch 4, accumulation 4.
Preserve original SIMS data, Chinese prompts, score range [-1,1], pseudo tokens 4,
generation tokens 4 and all router/trainer settings. Only replace frozen model,
tokenizer and hidden dimensions. Llama 3.2 uses the existing modern Transformers
environment and eager attention; Qwen/Llama 2 use the original environment.

Batch scripts: scripts/train_{qwen18,llama32,llama2}.slurm.
Output root: ../../outputs/sims_backbone_lr1e3_d00_sat3090_20260912.
Logs: <output>/<target>/slurm/<job-name>-<array-id>_<index>.{out,err}.
Results: <output>/<target>/task_<seed>/seed_<seed>/result.json.

SAT QoS allows at most eight submitted jobs including other experiments.
The lightweight detached scripts/submit_deferred.py process submits the nine
specified tasks as slots become available; it never trains on the login node.
Live submission ledger: <output>/submissions.json; dispatcher.log and dispatcher.pid
are in the same output directory. Array indices 0,1,2 map to 1111,1113,1115.
An exclusive lock and persisted submission intent prevent duplicate dispatch.
