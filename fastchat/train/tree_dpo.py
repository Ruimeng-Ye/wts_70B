import os
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, AutoConfig
from datasets import load_dataset, Dataset
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from fastchat.train.dpo_trainer import DPOMultiTrainer
from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template, get_model_adapter
from tqdm import tqdm

iteration = 1  

if iteration == 1:
    beta = 0.1  # 降低beta值，减少KL惩罚
    learning_rate = 5e-6  # 降低学习率，使训练更稳定
else:
    beta = 0.5
    learning_rate = 5e-7

max_length = 4096

training_args = TrainingArguments(
    output_dir="/home/bhui/ML/ruimeng/ETO-main/dpo_output/TreeDPO_llama13B_sciworld",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    save_steps=80,
    save_total_limit=5,
    learning_rate=learning_rate,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="constant_with_warmup",
    logging_steps=10,
    tf32=True,
    gradient_checkpointing=True,  
    max_grad_norm=0.5,
    # 添加一些额外的监控和调试参数
    logging_first_step=True,  # 记录第一步的指标
    report_to=None,   # 使用tensorboard记录训练过程
    fp16=True,                 # 使用混合精度训练
)

# 设置模型路径
base_model_path = "/home/bhui/ML/ruimeng/ETO-main/webshop_output/lora_sft_strong_llama2_web_v1/merged_model"
ref_model_path = "/home/bhui/ML/ruimeng/ETO-main/webshop_output/lora_sft_weak_llama2_web_v1/merged_model"

# 设置数据路径
data_path = "sciworld_dpo_pairs.json"

# 加载配置
policy_config = AutoConfig.from_pretrained(base_model_path)
orig_ctx_len = getattr(policy_config, "max_position_embeddings", None)
if orig_ctx_len and max_length > orig_ctx_len:
    scaling_factor = float(math.ceil(max_length / orig_ctx_len))
    policy_config.rope_scaling = {"type": "linear", "factor": scaling_factor}
policy_config.use_cache = False

# 加载模型
print("加载政策模型...")
policy_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    config=policy_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# 配置LoRA
lora_config = LoraConfig(
    r=128,
    lora_alpha=256,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

policy_model = get_peft_model(policy_model, lora_config)

# 参考模型
print("加载参考模型...")
ref_config = AutoConfig.from_pretrained(ref_model_path)
orig_ctx_len = getattr(ref_config, "max_position_embeddings", None)
if orig_ctx_len and max_length > orig_ctx_len:
    scaling_factor = float(math.ceil(max_length / orig_ctx_len))
    ref_config.rope_scaling = {"type": "linear", "factor": scaling_factor}
ref_config.use_cache = False

ref_model = AutoModelForCausalLM.from_pretrained(
    ref_model_path,
    config=ref_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

for param in ref_model.parameters():
    param.requires_grad = False

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    model_max_length=max_length,
    padding_side="right", 
    use_fast=False,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 打印模型和分词器信息（用于调试）
print(f"政策模型词表大小: {policy_model.config.vocab_size}")
print(f"参考模型词表大小: {ref_model.config.vocab_size}")
print(f"分词器词表大小: {len(tokenizer)}")

# 确保词表大小匹配
if policy_model.config.vocab_size != len(tokenizer):
    print(f"警告: 政策模型词表大小 ({policy_model.config.vocab_size}) 与分词器词表大小 ({len(tokenizer)}) 不一致")
    policy_model.resize_token_embeddings(len(tokenizer))
    print(f"已调整政策模型词表大小为: {policy_model.config.vocab_size}")

# 加载数据集
print("加载数据集...")
dataset = load_dataset("json", data_files=data_path)
print(f"数据集大小: {len(dataset['train'])}")

# 直接处理轨迹树数据，提取成功和失败动作
def extract_action_pairs(dataset):
    """直接从数据集中提取成功和失败动作对"""
    valid_pairs = []
    
    print(f"开始处理 {len(dataset)} 个样本...")
    for i, example in enumerate(tqdm(dataset)):
        try:
            # 提取关键动作
            success_action = example.get('success_action', '')
            failure_action = example.get('failure_action', '')
            
            # 检查动作是否有效
            if not success_action or not failure_action or success_action == failure_action:
                continue
                
            # 提取指令或使用默认指令
            instruction = example.get('instruction', 'You are web shopping.')
            
            # 从success_trajectory中提取状态，如果存在
            state = "Please decide what to do next."
            if 'success_trajectory' in example and 'sequence' in example['success_trajectory'] and example['success_trajectory']['sequence']:
                first_action = example['success_trajectory']['sequence'][0]
                if isinstance(first_action, dict) and 'state' in first_action:
                    state = first_action['state']
            
            # 将成功和失败动作添加到有效对列表
            valid_pairs.append({
                "instruction": instruction,
                "state": state,
                "success_action": success_action,
                "failure_action": failure_action
            })
            
        except Exception as e:
            if i % 100 == 0:
                print(f"样本 {i} 处理出错: {e}")
    
    print(f"从 {len(dataset)} 个样本中提取了 {len(valid_pairs)} 对有效动作")
    return valid_pairs

# 转换为DPO训练数据集
def convert_to_dpo_format(action_pairs):
    """转换动作对为DPO训练数据格式"""
    dpo_samples = []
    
    for pair in tqdm(action_pairs):
        try:
            # 构建对话
            system_prompt = pair["instruction"]
            context = f"Observation: {pair['state']}"
            
            # 提示部分
            prompt = f"{system_prompt}\n\n{context}"
            
            # 成功轨迹
            chosen = f"{system_prompt}\n\n{context}\n\nThought: I need to choose the best action.\nAction: {pair['success_action']}"
            
            # 失败轨迹
            rejected = f"{system_prompt}\n\n{context}\n\nThought: I might choose this action.\nAction: {pair['failure_action']}"
            
            # 对话的分词
            prompt_tokens = tokenizer(prompt, return_tensors="pt")
            chosen_tokens = tokenizer(chosen, return_tensors="pt", max_length=max_length, truncation=True)
            rejected_tokens = tokenizer(rejected, return_tensors="pt", max_length=max_length, truncation=True)
            
            # 创建标签
            chosen_labels = chosen_tokens.input_ids[0].clone()
            prompt_len = len(prompt_tokens.input_ids[0])
            chosen_labels[:prompt_len] = -100
            
            rejected_labels = rejected_tokens.input_ids[0].clone()
            rejected_labels[:prompt_len] = -100
            
            # 检查是否有有效的预测标签
            chosen_valid = (chosen_labels != -100).sum().item()
            rejected_valid = (rejected_labels != -100).sum().item()
            
            if chosen_valid > 0 and rejected_valid > 0:
                dpo_samples.append({
                    "prompt_input_ids": prompt_tokens.input_ids[0],
                    "prompt_attention_mask": prompt_tokens.attention_mask[0],
                    "chosen_input_ids": chosen_tokens.input_ids[0],
                    "chosen_attention_mask": chosen_tokens.attention_mask[0],
                    "chosen_labels": chosen_labels,
                    "rejected_input_ids": rejected_tokens.input_ids[0],
                    "rejected_attention_mask": rejected_tokens.attention_mask[0],
                    "rejected_labels": rejected_labels,
                })
                
        except Exception as e:
            print(f"转换DPO格式出错: {e}")
            continue
    
    print(f"成功创建 {len(dpo_samples)} 个DPO训练样本")
    return dpo_samples

# 先创建一些手动样本，作为备份
def create_manual_samples():
    """创建一组手动构建的训练样本，作为备份"""
    samples = []
    
    # 样本1: 搜索vs随机点击
    samples.append({
        "instruction": "You are web shopping. You need to choose the best action.",
        "state": "You are looking for a high-quality laptop under $1000.",
        "success_action": "search[laptop high quality under $1000]",
        "failure_action": "click[random link]"
    })
    
    # 样本2: 明确搜索vs模糊搜索
    samples.append({
        "instruction": "You are web shopping. You need to choose the best action.",
        "state": "You need to find organic whole milk.",
        "success_action": "search[organic whole milk]",
        "failure_action": "search[milk]"
    })
    
    # 样本3: 相关点击vs不相关点击
    samples.append({
        "instruction": "You are web shopping. You need to choose the best action.",
        "state": "You see a list of smartphones. You want the one with best camera.",
        "success_action": "click[iPhone 14 Pro Max]",
        "failure_action": "click[Nokia 3310]"
    })
    
    # 样本4: 查看详情vs立即购买
    samples.append({
        "instruction": "You are web shopping. You need to choose the best action.",
        "state": "You found an interesting product but want to know more details.",
        "success_action": "click[View Details]",
        "failure_action": "click[Buy Now]"
    })
    
    # 样本5: 选择合适尺寸vs随机尺寸
    samples.append({
        "instruction": "You are web shopping. You need to choose the best action.",
        "state": "You need to select a size for the shoes you want to buy. You wear size 9.",
        "success_action": "click[Size 9]",
        "failure_action": "click[Size 12]"
    })
    
    return samples

# 开始处理您的数据
print("从数据集中提取动作对...")
action_pairs = extract_action_pairs(dataset['train'])

# 如果没有足够的有效样本，添加手动样本
if len(action_pairs) < 5:
    print(f"有效样本数量太少 ({len(action_pairs)}), 添加手动样本...")
    manual_samples = create_manual_samples()
    action_pairs.extend(manual_samples)
    print(f"添加了 {len(manual_samples)} 个手动样本，总共有 {len(action_pairs)} 个样本")

# 转换为DPO训练格式
print("转换为DPO训练格式...")
dpo_samples = convert_to_dpo_format(action_pairs)

# 确保有足够的训练样本
if len(dpo_samples) < 5:
    print("警告: 有效样本数量仍然不足，训练可能不稳定")
    # 创建更多的副本以增加样本数量
    while len(dpo_samples) < 5:
        dpo_samples.extend(dpo_samples[:min(5, len(dpo_samples))])
    print(f"通过复制扩展到 {len(dpo_samples)} 个样本")

# 分析样本
if len(dpo_samples) > 0:
    sample = dpo_samples[0]
    print("\n样本分析:")
    chosen_tokens_count = (sample["chosen_labels"] != -100).sum().item()
    rejected_tokens_count = (sample["rejected_labels"] != -100).sum().item()
    print(f"需要预测的chosen token数量: {chosen_tokens_count}")
    print(f"需要预测的rejected token数量: {rejected_tokens_count}")
    
    if chosen_tokens_count > 0:
        mask_positions = (sample["chosen_labels"] != -100).nonzero(as_tuple=True)[0]
        if len(mask_positions) > 0:
            start_pos = mask_positions[0].item()
            end_pos = min(start_pos + 50, len(sample["chosen_input_ids"]))
            chosen_content = tokenizer.decode(sample["chosen_input_ids"][start_pos:end_pos])
            print(f"Chosen预测部分: {chosen_content}")
    
    if rejected_tokens_count > 0:
        mask_positions = (sample["rejected_labels"] != -100).nonzero(as_tuple=True)[0]
        if len(mask_positions) > 0:
            start_pos = mask_positions[0].item()
            end_pos = min(start_pos + 50, len(sample["rejected_input_ids"]))
            rejected_content = tokenizer.decode(sample["rejected_input_ids"][start_pos:end_pos])
            print(f"Rejected预测部分: {rejected_content}")

# 创建训练数据集
preprocessed_dataset = Dataset.from_list(dpo_samples)

# 计算并保存一些统计信息
with open(os.path.join(training_args.output_dir, "dataset_stats.txt"), "w") as f:
    f.write(f"原始数据集大小: {len(dataset['train'])}\n")
    f.write(f"提取的有效动作对数: {len(action_pairs)}\n")
    f.write(f"最终训练样本数: {len(dpo_samples)}\n")
    f.write("\n样本示例:\n")
    if len(dpo_samples) > 0:
        sample = dpo_samples[0]
        prompt = tokenizer.decode(sample["prompt_input_ids"])
        chosen = tokenizer.decode(sample["chosen_input_ids"])
        rejected = tokenizer.decode(sample["rejected_input_ids"])
        f.write(f"Prompt: {prompt}\n\n")
        f.write(f"Chosen: {chosen}\n\n")
        f.write(f"Rejected: {rejected}\n\n")

# 初始化 DPOMultiTrainer
trainer = DPOMultiTrainer(
    model=policy_model,
    ref_model=ref_model,
    args=training_args,
    beta=beta,
    train_dataset=preprocessed_dataset,
    tokenizer=tokenizer,
    max_length=max_length,
    max_prompt_length=512,
    max_target_length=3072,
    generate_during_eval=False,
)

# 检查第一批次的损失计算
def check_batch_loss(trainer, batch_size=2):
    """检查第一批次的损失计算是否正常"""
    print("\n检查第一批次的损失计算...")
    # 创建一个小批次
    mini_batch = {
        k: preprocessed_dataset[0:batch_size][k]
        for k in preprocessed_dataset[0:batch_size].keys()
    }
    
    # 将所有张量移动到GPU
    for k, v in mini_batch.items():
        if isinstance(v, torch.Tensor):
            mini_batch[k] = v.to(trainer.args.device)
    
    # 计算损失和指标
    try:
        with torch.no_grad():
            loss, metrics = trainer.get_batch_loss_metrics(trainer.model, mini_batch, train_eval="train")
        print(f"损失: {loss.item()}")
        print(f"指标: {metrics}")
        
        if abs(loss.item() - 0.6931) < 0.01 and all(abs(v) < 0.01 for v in metrics.values()):
            print("警告: 损失和指标都接近默认值，可能有问题!")
        else:
            print("损失计算正常，可以开始训练。")
    except Exception as e:
        print(f"计算损失时出错: {e}")
        import traceback
        traceback.print_exc()

# 检查损失计算
check_batch_loss(trainer, batch_size=2)

# 开始训练，支持从检查点恢复
print("开始训练...")
if os.path.exists(os.path.join(training_args.output_dir, "checkpoint-last")):
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()

# 保存 DPO LoRA 权重
dpo_lora_path = os.path.join(training_args.output_dir, "final_lora")
trainer.model.save_pretrained(dpo_lora_path)

print("DPO训练完成，LoRA权重已保存。")

# 将 DPO LoRA 权重合并到政策模型
print("开始合并 DPO LoRA 权重到政策模型...")
# 加载政策模型的基础模型
base_policy_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    config=policy_config,
    torch_dtype=torch.float16,
)

# 加载 LoRA 权重到政策模型
peft_model = PeftModel.from_pretrained(base_policy_model, dpo_lora_path)
merged_model = peft_model.merge_and_unload()

# 保存最终合并后的模型
final_model_path = os.path.join(training_args.output_dir, "final_merged_model")
merged_model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)

print(f"合并完成。最终模型已保存至 {final_model_path}")