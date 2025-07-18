from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=.1, top_p=.95)
llm = LLM(model="/work/nvme/betg/mshtepel/models/Qwen/Qwen2.5-1.5B-Instruct")
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# destroy_process_group()