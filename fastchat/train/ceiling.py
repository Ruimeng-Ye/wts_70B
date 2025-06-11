# 这个结果非常好   是对于模型自身的DPO的优化
import os
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, AutoConfig
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from fastchat.train.dpo_trainer import DPOMultiTrainer
from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template, get_model_adapter

# 根据迭代次数调整 beta 和学习率
iteration = 1  # 根据需要设置迭代次数

if iteration == 1:
    beta = 0.1
    learning_rate = 1e-6
else:
    beta = 0.5
    learning_rate = 5e-7

# 设置最大长度
max_length = 4096

# 设置训练参数
training_args = TrainingArguments(
    output_dir="/home/bhui/ML/ruimeng/ETO-main/webshop/llama8B_webshop_wts_ref_itself_ceiling",
    num_train_epochs=3,
    per_device_train_batch_size=1,  
    gradient_accumulation_steps=32,  
    save_steps=500,
    save_total_limit=5,
    learning_rate=learning_rate,
    weight_decay=0.0,
    warmup_ratio=0.05,
    lr_scheduler_type="constant_with_warmup",
    logging_steps=10,
    tf32=True,
    gradient_checkpointing=True,  # 启用梯度检查点
    max_grad_norm=0.5,  # 设置梯度的最大规范化值
)

# Adjust RoPE scaling for longer context lengths
base_model_path = "/home/bhui/ML/ruimeng/ETO-main/llama8B_webshop_output/lora_sft_strong_model/merged_model"  # 使用适合你的基础模型路径
config = AutoConfig.from_pretrained(base_model_path)
orig_ctx_len = getattr(config, "max_position_embeddings", None)
if orig_ctx_len and max_length > orig_ctx_len:
    scaling_factor = float(math.ceil(max_length / orig_ctx_len))
    config.rope_scaling = {"type": "linear", "factor": scaling_factor}
config.use_cache = False

# 加载分词器并调整填充标记 - 先加载tokenizer以便后续能够正确调整模型嵌入大小
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path, 
    model_max_length=max_length,
    padding_side="right",  # 根据需要调整
    use_fast=False,
)

# 确保tokenizer有pad_token和pad_token_id
if tokenizer.pad_token is None:
    # 为了LLM模型，最好将pad_token设为eos_token
    tokenizer.pad_token = tokenizer.eos_token
    print(f"已设置 pad_token = eos_token: {tokenizer.pad_token}")

# 加载基础模型（用于训练的policy model）
policy_model = AutoModelForCausalLM.from_pretrained(
    base_model_path, 
    config=config,
    device_map="auto", 
    torch_dtype=torch.float16,
)

# 如果tokenizer词表大小发生变化，需要调整模型的嵌入层
if len(tokenizer) != policy_model.config.vocab_size:
    policy_model.resize_token_embeddings(len(tokenizer))
    print(f"已调整policy_model词汇表大小为: {len(tokenizer)}")

# 为policy model配置LoRA
lora_config = LoraConfig(
    r=128,
    lora_alpha=128,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# 将LoRA应用到policy model
policy_model = get_peft_model(policy_model, lora_config)

# 加载参考模型（已包含SFT LoRA权重）
ref_model_path = "/home/bhui/ML/ruimeng/ETO-main/llama8B_webshop_output/lora_sft_strong_model/merged_model"
ref_model = AutoModelForCausalLM.from_pretrained(
    ref_model_path, 
    config=config,
    device_map="auto", 
    torch_dtype=torch.float16,
)

# 如果tokenizer词表大小发生变化，需要调整ref_model的嵌入层
if len(tokenizer) != ref_model.config.vocab_size:
    ref_model.resize_token_embeddings(len(tokenizer))
    print(f"已调整ref_model词汇表大小为: {len(tokenizer)}")

# 将参考模型设置为不可训练
for param in ref_model.parameters():
    param.requires_grad = False

# 打印tokenizer信息，以便调试
# print(f"Tokenizer pad_token: '{tokenizer.pad_token}'")
# print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
# print(f"Tokenizer eos_token: '{tokenizer.eos_token}'")
# print(f"Tokenizer eos_token_id: {tokenizer.eos_token_id}")

# 加载和预处理数据集
data_path = "data_pm/ceiling_8B_dpo.json"
dataset = load_dataset("json", data_files=data_path)

IGNORE_TOKEN_ID = -100  # LabelSmoother.ignore_index

def mask_labels(conversation, target, tokenizer, conv):
    if conv.sep_style == SeparatorStyle.ADD_COLON_TWO:
        sep = conv.sep + conv.roles[1] + ": "
    elif conv.sep_style == SeparatorStyle.LLAMA2:
        sep = conv.sep + conv.roles[1] + " "
    elif conv.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
        sep = conv.sep + conv.roles[1] + ":"
    elif conv.sep_style == SeparatorStyle.NO_COLON_SINGLE:
        sep = conv.sep + conv.roles[1]
    elif conv.sep_style == SeparatorStyle.ADD_COLON_TWO_NO_SPACE:
        sep = conv.sep + conv.roles[1] + ":"
    else:
        # 默认行为：使用通用分隔符
        sep = conv.sep + conv.roles[1]
    
    # 使用pad_token_id计算total_len
    total_len = int(target.ne(tokenizer.pad_token_id).sum())

    turns = conversation.split(conv.sep2)
    cur_len = 1
    target[:cur_len] = IGNORE_TOKEN_ID
    for i, turn in enumerate(turns):
        if turn == "":
            break

        turn_len = len(tokenizer(turn).input_ids) - 1

        parts = turn.split(sep, 1)  # 只分割一次，以防文本中包含分隔符

        if len(parts) != 2:
            # 如果分割失败，跳过这个 turn
            cur_len += turn_len
            continue

        instruction = parts[0] + sep
        response = parts[1]
        
        instruction_len = len(tokenizer(instruction).input_ids) - 1
        response_len = len(tokenizer(response).input_ids)

        # 忽略用户的指令部分
        target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
        cur_len += instruction_len + response_len

        # 增加 turn sep 的长度
        if conv.sep2:
            cur_len += len(tokenizer(conv.sep2).input_ids)

    target[cur_len:] = IGNORE_TOKEN_ID

    return target

def preprocess_function(example):
    try:
        conv = get_model_adapter(base_model_path).get_default_conv_template(base_model_path)
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # 应用提示模板
        conv.messages = []
        for j, sentence in enumerate(example['prompt']):
            role = roles[sentence["from"]]
            conv.append_message(role, sentence["value"])
        prompt = conv.get_prompt()

        conv.messages = []
        for j, sentence in enumerate(example['prompt'] + example['chosen']):
            role = roles[sentence["from"]]
            conv.append_message(role, sentence["value"])
        chosen = conv.get_prompt()

        conv.messages = []
        for j, sentence in enumerate(example['prompt'] + example['rejected']):
            role = roles[sentence["from"]]
            conv.append_message(role, sentence["value"])
        rejected = conv.get_prompt()

        # 对话的分词
        prompt_tokens = tokenizer(prompt, return_tensors="pt")

        chosen_tokens = tokenizer(chosen, return_tensors="pt", max_length=max_length, truncation=True)
        chosen_labels = chosen_tokens.input_ids[0].clone()
        chosen_labels = mask_labels(chosen, chosen_labels, tokenizer, conv)
        chosen_labels[:len(prompt_tokens['input_ids'][0])] = IGNORE_TOKEN_ID

        rejected_tokens = tokenizer(rejected, return_tensors="pt", max_length=max_length, truncation=True)
        rejected_labels = rejected_tokens.input_ids[0].clone()
        rejected_labels = mask_labels(rejected, rejected_labels, tokenizer, conv)
        rejected_labels[:len(prompt_tokens['input_ids'][0])] = IGNORE_TOKEN_ID

        return {
            "prompt_input_ids": prompt_tokens['input_ids'][0],
            "prompt_attention_mask": prompt_tokens['attention_mask'][0],
            "chosen_input_ids": chosen_tokens['input_ids'][0],
            "chosen_attention_mask": chosen_tokens['attention_mask'][0],
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_tokens['input_ids'][0],
            "rejected_attention_mask": rejected_tokens['attention_mask'][0],
            "rejected_labels": rejected_labels,
        }
    except Exception as e:
        print(f"处理样本时出错: {e}")
        # 打印出样本以帮助调试
        import traceback
        traceback.print_exc()
        print(f"样本内容: {example}")
        raise e

# 预处理数据集
print("开始预处理数据集...")
preprocessed_dataset = dataset['train'].map(
    preprocess_function, 
    remove_columns=dataset['train'].column_names,
    num_proc=1,  # 减少并行处理，避免潜在的冲突
    desc="Preprocessing dataset"
)
print(f"预处理完成，样本数: {len(preprocessed_dataset)}")

# 初始化DPOMultiTrainer，确保使用同一个tokenizer
trainer = DPOMultiTrainer(
    model=policy_model,
    ref_model=ref_model,
    args=training_args,
    beta=beta,
    train_dataset=preprocessed_dataset,
    tokenizer=tokenizer,  # 传递已正确设置pad_token的tokenizer
    max_length=max_length,
    max_prompt_length=512,
    max_target_length=3072,
    generate_during_eval=False,  # 训练评估时不生成文本
    # 确保传入pad_token_id
    label_pad_token_id=IGNORE_TOKEN_ID,
    padding_value=tokenizer.pad_token_id,
)

# 开始训练，支持从检查点恢复
print("开始训练...")
if os.path.exists(os.path.join(training_args.output_dir, "checkpoint-last")):
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()

# 保存DPO LoRA权重
dpo_lora_path = os.path.join(training_args.output_dir, "final_lora")
trainer.model.save_pretrained(dpo_lora_path)

print("DPO训练完成，LoRA权重已保存。")

# 将DPO LoRA权重合并到参考模型
print("开始合并DPO LoRA权重到参考模型...")
ref_model = PeftModel.from_pretrained(ref_model, dpo_lora_path)
merged_model = ref_model.merge_and_unload()

# 保存最终合并后的模型
final_model_path = os.path.join(training_args.output_dir, "final_merged_model")
merged_model.save_pretrained(final_model_path)

print(f"合并完成。最终模型已保存至 {final_model_path}")


# # 这个结果非常好   是对于模型自身的DPO的优化
# import os
# import math
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, AutoConfig
# from datasets import load_dataset
# from peft import get_peft_model, LoraConfig, TaskType, PeftModel
# from fastchat.train.dpo_trainer import DPOMultiTrainer
# from fastchat.conversation import SeparatorStyle
# from fastchat.model.model_adapter import get_conversation_template, get_model_adapter

# # 根据迭代次数调整 beta 和学习率
# iteration = 1  # 根据需要设置迭代次数

# if iteration == 1:
#     beta = 0.1
#     learning_rate = 1e-6
# else:
#     beta = 0.5
#     learning_rate = 5e-7

# # 设置最大长度
# max_length = 4096

# # 设置训练参数
# training_args = TrainingArguments(
#     output_dir="/home/bhui/ML/ruimeng/ETO-main/webshop/llama8B_webshop_wts_ref_itself",
#     num_train_epochs=3,
#     per_device_train_batch_size=1,  
#     gradient_accumulation_steps=32,  
#     save_steps=500,
#     save_total_limit=5,
#     learning_rate=learning_rate,
#     weight_decay=0.0,
#     warmup_ratio=0.05,
#     lr_scheduler_type="constant_with_warmup",
#     logging_steps=10,
#     tf32=True,
#     gradient_checkpointing=True,  # 启用梯度检查点
#     max_grad_norm=0.5,  # 设置梯度的最大规范化值
# )

# # Adjust RoPE scaling for longer context lengths
# base_model_path = "/home/bhui/ML/ruimeng/ETO-main/llama8B_webshop_output/lora_sft_strong_model/merged_model"  # 使用适合你的基础模型路径
# config = AutoConfig.from_pretrained(base_model_path)
# orig_ctx_len = getattr(config, "max_position_embeddings", None)
# if orig_ctx_len and max_length > orig_ctx_len:
#     scaling_factor = float(math.ceil(max_length / orig_ctx_len))
#     config.rope_scaling = {"type": "linear", "factor": scaling_factor}
# config.use_cache = False

# # 加载基础模型（用于训练的policy model）
# policy_model = AutoModelForCausalLM.from_pretrained(
#     base_model_path, 
#     config=config,
#     device_map="auto", 
#     torch_dtype=torch.float16,
# )

# # 为policy model配置LoRA
# lora_config = LoraConfig(
#     r=128,
#     lora_alpha=128,
#     target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
#     lora_dropout=0.1,
#     bias="none",
#     task_type=TaskType.CAUSAL_LM
# )

# # 将LoRA应用到policy model
# policy_model = get_peft_model(policy_model, lora_config)

# # 加载参考模型（已包含SFT LoRA权重）
# ref_model_path = "/home/bhui/ML/ruimeng/ETO-main/llama8B_webshop_output/lora_sft_strong_model/merged_model"
# ref_model = AutoModelForCausalLM.from_pretrained(
#     ref_model_path, 
#     config=config,
#     device_map="auto", 
#     torch_dtype=torch.float16,
# )

# # 将参考模型设置为不可训练
# for param in ref_model.parameters():
#     param.requires_grad = False

# # 加载分词器并调整填充标记
# tokenizer = AutoTokenizer.from_pretrained(
#     ref_model_path, 
#     model_max_length=max_length,
#     padding_side="right",  # 根据需要调整
#     use_fast=False,
# )
# if tokenizer.pad_token != tokenizer.unk_token:
#     tokenizer.pad_token = tokenizer.unk_token

# # 加载和预处理数据集
# data_path = "/home/bhui/ML/ruimeng/ETO-main/data_pm/8B_dpo.json"
# dataset = load_dataset("json", data_files=data_path)

# IGNORE_TOKEN_ID = -100  # LabelSmoother.ignore_index

# def mask_labels(conversation, target, tokenizer, conv):
#     if conv.sep_style == SeparatorStyle.ADD_COLON_TWO:
#         sep = conv.sep + conv.roles[1] + ": "
#     elif conv.sep_style == SeparatorStyle.LLAMA2:
#         sep = conv.sep + conv.roles[1] + " "
#     elif conv.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
#         sep = conv.sep + conv.roles[1] + ":"
#     elif conv.sep_style == SeparatorStyle.NO_COLON_SINGLE:
#         sep = conv.sep + conv.roles[1]
#     elif conv.sep_style == SeparatorStyle.ADD_COLON_TWO_NO_SPACE:
#         sep = conv.sep + conv.roles[1] + ":"
#     else:
#         # 默认行为：使用通用分隔符
#         sep = conv.sep + conv.roles[1]
    
#     total_len = int(target.ne(tokenizer.pad_token_id).sum())

#     turns = conversation.split(conv.sep2)
#     cur_len = 1
#     target[:cur_len] = IGNORE_TOKEN_ID
#     for i, turn in enumerate(turns):
#         if turn == "":
#             break

#         turn_len = len(tokenizer(turn).input_ids) - 1

#         parts = turn.split(sep, 1)  # 只分割一次，以防文本中包含分隔符

#         if len(parts) != 2:
#             # 如果分割失败，跳过这个 turn
#             cur_len += turn_len
#             continue

#         instruction = parts[0] + sep
#         response = parts[1]
        
#         instruction_len = len(tokenizer(instruction).input_ids) - 1
#         response_len = len(tokenizer(response).input_ids)

#         # 忽略用户的指令部分
#         target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
#         cur_len += instruction_len + response_len

#         # 增加 turn sep 的长度
#         if conv.sep2:
#             cur_len += len(tokenizer(conv.sep2).input_ids)

#     target[cur_len:] = IGNORE_TOKEN_ID

#     return target

# def preprocess_function(example):
#     conv = get_model_adapter(base_model_path).get_default_conv_template(base_model_path)
#     roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

#     # 应用提示模板
#     conv.messages = []
#     for j, sentence in enumerate(example['prompt']):
#         role = roles[sentence["from"]]
#         conv.append_message(role, sentence["value"])
#     prompt = conv.get_prompt()

#     conv.messages = []
#     for j, sentence in enumerate(example['prompt'] + example['chosen']):
#         role = roles[sentence["from"]]
#         conv.append_message(role, sentence["value"])
#     chosen = conv.get_prompt()

#     conv.messages = []
#     for j, sentence in enumerate(example['prompt'] + example['rejected']):
#         role = roles[sentence["from"]]
#         conv.append_message(role, sentence["value"])
#     rejected = conv.get_prompt()

#     # 对话的分词
#     prompt_tokens = tokenizer(prompt, return_tensors="pt")

#     chosen_tokens = tokenizer(chosen, return_tensors="pt", max_length=max_length, truncation=True)
#     chosen_labels = chosen_tokens.input_ids[0].clone()
#     chosen_labels = mask_labels(chosen, chosen_labels, tokenizer, conv)
#     chosen_labels[:len(prompt_tokens['input_ids'][0])] = IGNORE_TOKEN_ID

#     rejected_tokens = tokenizer(rejected, return_tensors="pt", max_length=max_length, truncation=True)
#     rejected_labels = rejected_tokens.input_ids[0].clone()
#     rejected_labels = mask_labels(rejected, rejected_labels, tokenizer, conv)
#     rejected_labels[:len(prompt_tokens['input_ids'][0])] = IGNORE_TOKEN_ID

#     return {
#         "prompt_input_ids": prompt_tokens['input_ids'][0],
#         "prompt_attention_mask": prompt_tokens['attention_mask'][0],
#         "chosen_input_ids": chosen_tokens['input_ids'][0],
#         "chosen_attention_mask": chosen_tokens['attention_mask'][0],
#         "chosen_labels": chosen_labels,
#         "rejected_input_ids": rejected_tokens['input_ids'][0],
#         "rejected_attention_mask": rejected_tokens['attention_mask'][0],
#         "rejected_labels": rejected_labels,
#     }

# # 预处理数据集
# preprocessed_dataset = dataset['train'].map(preprocess_function, remove_columns=dataset['train'].column_names)

# # 初始化DPOMultiTrainer
# trainer = DPOMultiTrainer(
#     model=policy_model,
#     ref_model=ref_model,
#     args=training_args,
#     beta=beta,
#     train_dataset=preprocessed_dataset,
#     tokenizer=tokenizer,
#     max_length=max_length,
#     max_prompt_length=512,
#     max_target_length=3072,
#     generate_during_eval=False,  # 训练评估时不生成文本
# )

# # 开始训练，支持从检查点恢复
# if os.path.exists(os.path.join(training_args.output_dir, "checkpoint-last")):
#     trainer.train(resume_from_checkpoint=True)
# else:
#     trainer.train()

# # 保存DPO LoRA权重
# dpo_lora_path = os.path.join(training_args.output_dir, "final_lora")
# trainer.model.save_pretrained(dpo_lora_path)

# print("DPO训练完成，LoRA权重已保存。")

# # 将DPO LoRA权重合并到参考模型
# print("开始合并DPO LoRA权重到参考模型...")
# ref_model = PeftModel.from_pretrained(ref_model, dpo_lora_path)
# merged_model = ref_model.merge_and_unload()

# # 保存最终合并后的模型
# final_model_path = os.path.join(training_args.output_dir, "final_merged_model")
# merged_model.save_pretrained(final_model_path)

# print(f"合并完成。最终模型已保存至 {final_model_path}")