from vllm import LLM, SamplingParams


MODEL_PATH = "./models/Qwen3-VL-8B-Instruct"


llm = LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    tensor_parallel_size=2,
    dtype="float16",
    gpu_memory_utilization=0.85,
    cpu_offload_gb=4.0,
    max_model_len=2048,
    max_num_seqs=1,
    limit_mm_per_prompt={"image": 1, "video": 0},
)

messages = [
    {
        "role": "user",
        "content": "hello, how do you do",
    }
]

outputs = llm.chat(
    messages,
    sampling_params=SamplingParams(
        temperature=0.0,
        max_tokens=64,
    ),
)

print(outputs[0].outputs[0].text.strip())
