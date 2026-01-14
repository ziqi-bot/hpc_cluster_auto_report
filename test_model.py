# from transformers import AutoTokenizer, AutoModelForCausalLM

# MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir="/data/researchHome/pdou/hpc_cluster_report/model_cache")
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     cache_dir="/data/researchHome/pdou/hpc_cluster_report/model_cache"
# )

# print("Llama 3.1-8B-Instruct model loaded successfully!")


import transformers
import torch

model_id = "meta-llama/Llama-3.3-70B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

# 将多个消息拼接成一个完整的 prompt
prompt = ""
for msg in messages:
    role = msg["role"]
    content = msg["content"]
    if role == "system":
        # 可以在开头加上系统指令，让模型知道需要遵守这个设定
        prompt += f"{content}\n"
    elif role == "user":
        prompt += f"User: {content}\n"
    # 若有assistant角色消息，可考虑加入 Assistant: 前缀，如下：
    # elif role == "assistant":
    #     prompt += f"Assistant: {content}\n"

# 对 prompt 进行推断
outputs = pipeline(
    prompt,
    max_new_tokens=256,
    # 如果是Instrcut-tuned模型，可能需要bos_token作为开头，具体需看模型文档，这里仅示例
)

# 打印完整生成的文本
print(outputs[0]["generated_text"])
