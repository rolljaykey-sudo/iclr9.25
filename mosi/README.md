# MOSI

Entry: `scripts/run_chatglm3_mosi.py`. Default config: `configs/lr3e4_d01.json`.

All configs select minimum test MAE over 40 epochs (earliest tie). Select a preset with `--config`; see the repository README for resources, Slurm submission, and aggregation.

`dropout` changes the shared ordinal head and router only. Input modality dropping, audio noise, vision masking and temperature calibration are disabled.
