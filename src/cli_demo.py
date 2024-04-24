from llmtuner import ChatModel
from llmtuner.extras.misc import torch_gc
import json

try:
    import platform

    if platform.system() != "Windows":
        import readline  # noqa: F401
except ImportError:
    print("Install `readline` for a better experience.")


def main():
    chat_model = ChatModel()
    messages = []
    print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")
    with open("/home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/data/eval_all.json", "r") as f:
        tasks = json.load(f)
        predicts = []
        for task in tasks:
            prompt = task["system"]+task["instruction"]+task["input"]
            try:
                query = prompt
            except UnicodeDecodeError:
                print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
                continue
            except Exception:
                raise

            if query.strip() == "exit":
                break

            if query.strip() == "clear":
                messages = []
                torch_gc()
                print("History has been removed.")
                continue
            messages.append({"role": "user", "content": query})
            print("Assistant: ", end="", flush=True)

            response = ""
            for new_text in chat_model.stream_chat(messages):
                print(new_text, end="", flush=True)
                response += new_text
            print()
            messages.append({"role": "assistant", "content": response})
            predicts.append({"label": "", "predict": f"{response}"})
            messages = []
            torch_gc()
    import jsonlines
    output_jsonl = "/home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/generated_predictions.jsonl"
    for predict in predicts:
        with jsonlines.open(output_jsonl, 'a') as writer:
            writer.write(predict)
if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=4 python src/export_model.py \
#     --model_name_or_path /home/zhangtaiyan/.cache/modelscope/hub/AI-ModelScope/Mixtral-8x7B-v0___1 \
#     --adapter_name_or_path /home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/save/Mistral-8x7B/pt/sft/lora/2024-03-30-21-14-46-wo_A-5e-5 \
#     --template default \
#     --finetuning_type lora \
#     --export_dir /home/zhangtaiyan/workspace/comp/my_finetune/LLaMA-Factory/save/Mistral-8x7B/pt/sft/lora/2024-03-30-21-14-46-wo_A-5e-5/export \
#     --export_size 2 \
#     --export_legacy_format False