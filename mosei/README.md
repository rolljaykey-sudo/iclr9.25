# ChatGLM3 · MOSEI

入口：`scripts/run_chatglm3_mosei.py`。学习率（适配器 / 诊断头与路由器）为 `0.0001 / 0.0001`，
路由器与诊断头 dropout 为 `0.1`，种子为 `[1111, 1113, 1115]`。
训练 40 轮，按测试集 `Non0_F1_score` 最大值选模，并列时保留最早轮次；不做温度校准。

源码来源：`mse router new conflict/revisions/mosei_lr1e4_testf1_20260910`。
已提取 ChatGLM3 专用实现，训练、损失及选模方法沿用源版本。
输出默认写入本目录的 `outputs/`，数据、模型和上游 ChatGLM3 组件仍使用外部资源。
运行和检查方法见仓库根目录 README。
