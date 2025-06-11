#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用DPO（Direct Preference Optimization）配合LoRA进行模型训练
基础模型：经过SFT的LLaMA-3 8B
参考模型：经过SFT的LLaMA-2 13B
"""

import os
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import logging
import json
import datasets
from datasets import load_dataset
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from peft import (
    LoraConfig,
    PeftModel,
    PeftConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import DPOTrainer, DPOConfig

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

@dataclass
class ScriptArguments:
    """
    DPO + LoRA训练的参数配置
    """
    # 模型相关参数
    base_model_name_or_path: str = field(
        metadata={"help": "SFT后的基础模型路径"}
    )
    reference_model_name_or_path: str = field(
        metadata={"help": "SFT后的参考模型路径"}
    )
    
    # 数据集相关参数
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "数据集名称或本地数据集路径"}
    )
    dataset_text_field: str = field(
        default="text", metadata={"help": "数据集中文本字段名称"}
    )
    
    # LoRA相关参数
    lora_r: int = field(default=16, metadata={"help": "LoRA适配器的秩"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha参数"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout概率"})
    
    # 输出与保存相关参数
    output_dir: str = field(default="./output", metadata={"help": "输出目录"})
    
    # 训练相关参数
    num_train_epochs: float = field(default=3.0, metadata={"help": "训练轮数"})
    per_device_train_batch_size: int = field(default=4, metadata={"help": "每个设备的训练批次大小"})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "梯度累积步数"})
    learning_rate: float = field(default=5e-5, metadata={"help": "学习率"})
    max_length: int = field(default=512, metadata={"help": "输入序列的最大长度"})
    logging_steps: int = field(default=10, metadata={"help": "日志记录间隔"})
    save_steps: int = field(default=100, metadata={"help": "模型保存间隔"})
    beta: float = field(default=0.1, metadata={"help": "DPO中的beta参数"})
    seed: int = field(default=42, metadata={"help": "随机种子"})

def prepare_dataset(args, tokenizer):
    """准备数据集格式，确保符合DPO训练所需格式"""
    
    if args.dataset_name.endswith('.json'):
        # 加载本地JSON数据集
        with open(args.dataset_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
        dataset = datasets.Dataset.from_dict({
            'prompt': [item['prompt'] for item in data],
            'chosen': [item['chosen'] for item in data],
            'rejected': [item['rejected'] for item in data]
        })
    else:
        # 尝试从Hugging Face加载数据集
        dataset = load_dataset(args.dataset_name)
        if 'train' in dataset:
            dataset = dataset['train']
    
    logger.info(f"加载了 {len(dataset)} 条训练数据")
    
    # 简单检查数据集格式是否符合DPO要求
    required_columns = ['prompt', 'chosen', 'rejected']
    for col in required_columns:
        if col not in dataset.column_names:
            raise ValueError(f"数据集缺少必需的列 '{col}'")
    
    return dataset

def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    set_seed(args.seed)
    
    # 加载基础模型和参考模型的tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基础模型
    logger.info(f"加载基础模型: {args.base_model_name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # 加载参考模型
    logger.info(f"加载参考模型: {args.reference_model_name_or_path}")
    reference_model = AutoModelForCausalLM.from_pretrained(
        args.reference_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # 配置LoRA
    logger.info("配置LoRA适配器")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    # 使用LoRA配置基础模型
    base_model = get_peft_model(base_model, peft_config)
    base_model.print_trainable_parameters()
    
    # 准备训练数据集
    logger.info("准备数据集")
    dataset = prepare_dataset(args, tokenizer)
    
    # 配置DPO训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        seed=args.seed,
        data_seed=args.seed,
        bf16=True,  # 使用bfloat16精度
        remove_unused_columns=False,
    )
    
    # 创建 DPO 配置
    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        num_train_epochs=args.num_train_epochs,
        # beta 参数单独传递
        seed=args.seed,
        bf16=True,
        remove_unused_columns=False,
        save_total_limit=3,
    )
    # 初始化 DPO 训练器
    
    trainer = DPOTrainer(
        model=base_model,
        ref_model=reference_model,
        beta=args.beta,
        args=dpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_prompt_length=args.max_length,
        peft_config=peft_config,  # 传递 peft_config
        force_use_ref_model=True  # 添加这个参数以明确指定使用ref_model
    )
    
    # 开始训练
    logger.info("开始DPO训练")
    trainer.train()
    
    # 保存LoRA适配器权重
    lora_output_dir = os.path.join(args.output_dir, "lora_weights")
    logger.info(f"保存LoRA权重到 {lora_output_dir}")
    trainer.model.save_pretrained(lora_output_dir)
    tokenizer.save_pretrained(lora_output_dir)
    
    # 合并LoRA权重到基础模型并保存
    logger.info("合并LoRA权重到基础模型")
    # 加载原始基础模型（不带LoRA）
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # 加载训练好的LoRA权重
    model_with_lora = PeftModel.from_pretrained(base_model, lora_output_dir)
    
    # 合并模型权重
    merged_model = model_with_lora.merge_and_unload()
    
    # 保存合并后的模型
    merged_output_dir = os.path.join(args.output_dir, "merged_model")
    logger.info(f"保存合并后的模型到 {merged_output_dir}")
    merged_model.save_pretrained(merged_output_dir)
    tokenizer.save_pretrained(merged_output_dir)
    
    logger.info("训练完成!")

if __name__ == "__main__":
    main()