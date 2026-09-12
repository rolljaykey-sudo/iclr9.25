#!/usr/bin/env python3
# SIMS 实验配置入口：复用通用 ChatGLM3 流程，设置本版本的种子、训练参数或输出位置。
"""ChatGLM3 SIMS 入口：沿用本数据集参数，按测试集 MAE 最小值 选择检查点。"""
import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from config_utils import load_spec, dataset_path

import chatglm_setup as backbone



# 以基线入口为基础装配当前实验的配置、输出目录、种子和训练参数。
def configure():
    SPEC_PATH, SPEC = load_spec("lr1e3_d00.json")
    backbone.configure_harness()
    h = backbone.harness
    h.SEEDS = tuple(SPEC['seeds'])
    h.DEFAULT_DATASET = dataset_path(SPEC, h.PROJECT_DIR)
    h.CONFIG_PATH = SPEC_PATH
    h.DEFAULT_OUTPUT_ROOT = h.PROJECT_DIR / "outputs" / SPEC_PATH.stem
    h.EXPECTED_SPLITS = SPEC['splits']
    h.EXPECTED_DATASET_SIZE = SPEC['dataset_size']
    h.EXPECTED_DATASET_SHA256 = SPEC['dataset_sha256']
    h.ROUTER_SOURCE_FILES += (Path(__file__).resolve(), SPEC_PATH, Path(__file__).resolve().parent / "config_utils.py")

    # 在上游配置基础上设置当前实验的数据、骨干维度、批量与路由参数。
    def build_config(args, microbatch, accumulation):
        import torch
        sys.path.insert(0, str(h.UPSTREAM_DIR))
        from config.config_regression import ConfigRegression
        base = argparse.Namespace(
            is_tune=False, tune_mode=False, train_mode='regression',
            modelName='cmcm', datasetName='simsv2',
            root_dataset_dir=str(args.dataset_path.resolve().parent),
            num_workers=args.num_workers, model_save_dir=str(args.output_root/'unused'),
            res_save_dir=str(args.output_root/'unused'),
            pretrain_LM=str(args.model_path.resolve()), gpu_ids=[0],
        )
        c = ConfigRegression(base).get_config()
        c.modelName = 'mse_router'
        c.datasetName = 'sims'
        c.dataPath = str(args.dataset_path.resolve())
        c.pretrain_LM = str(args.model_path.resolve())
        c.device = torch.device('cuda:0')
        c.batch_size = microbatch
        c.update_epochs = accumulation
        c.seq_lens = tuple(SPEC['sequence_lengths'])
        c.feature_dims = tuple(SPEC['router_feature_dims'])
        c.train_samples = SPEC['splits']['train']
        c.language = 'cn'
        c.H = 1.0
        c.task_specific_prompt = SPEC['task_specific_prompt']
        c.diagnostic_prompt = SPEC['diagnostic_prompt']
        c.router_hidden_size = 4096
        c.router_aux_weight = 0.3
        c.router_variant = args.router_variant
        c.router_dropout = SPEC['dropout']
        c.router_backbone = 'chatglm3'
        c.router_upstream_dir = str(h.UPSTREAM_DIR)
        c.gradient_checkpointing = True
        c.router_architecture = h.ARCHITECTURE
        c.router_conflict_metric = h.CONFLICT_METRIC
        return c

    h.build_config = build_config
    import mse_router.trainer as trainer
    # 覆盖基类中当前实验特有的训练参数，其余设置保持继承。
    @dataclass(frozen=True)
    class SimsSettings(trainer.TrainingSettings):
        adapter_lr: float = SPEC['adapter_lr']
        head_router_lr: float = SPEC['head_router_lr']
        stage1_max_epochs: int = 40
        stage1_patience: int = 41
        quality_gate_epoch: int = 0
    trainer.TrainingSettings = SimsSettings
    return h

if __name__ == '__main__':
    configure().main()
