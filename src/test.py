from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from huggingface_hub import snapshot_download

sql_lora_path = "/home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/save/Mistral-8x7B/pt/sft/lora/2024-03-30-21-14-46-wo_A-5e-5/"
llm = LLM(model="/home/zhangtaiyan/.cache/modelscope/hub/AI-ModelScope/Mixtral-8x7B-v0___1", tensor_parallel_size=4, enable_lora=True)  # Name or path of your model
sampling_params = SamplingParams(
    temperature=0.5,
    top_p=0.5,
    max_tokens=256,
)
import json
outputs = []
with open("/home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/data/eval_all.json", "r") as f:
    tasks = json.load(f)
    for task in tasks:
        prompt = task["system"]+task["instruction"]+task["input"]
        output = llm.generate(
            prompt,
            sampling_params,
            lora_request=LoRARequest("adapter_model",1, sql_lora_path)
        )
        import pdb; pdb.set_trace()
        outputs.append(output)
print(outputs)
