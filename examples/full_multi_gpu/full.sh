cd MY_PATH/comp/my_finetune/LLaMA-Factory/examples/full_multi_gpu
finetuning_type='full'
current_time=$(date +"%Y-%m-%d-%H-%M-%S")
# name_rule: save/${model_name}-${pretrain}(optional)-{sft}(optional)-${finetuning_type}-${current_time}-${dataset}-${lr}-${top_p&temperature}(optional)
model_name_or_path="MY_PATH/BioMistral-7B"
stage1='sft'
dataset='wo_A'
port="25674"
lr=1e-6
gpus="localhost:0,1,2,3"
eval_device="0"
full_sft_output_dir="MY_PATH/comp/my_finetune/LLaMA-Factory/save/BioMistral-7B/pt/sft/${finetuning_type}/${current_time}-${dataset}-${lr}"
step=20
val_size=0.02
deepspeed --master_port ${port} --include=${gpus}  ../../src/train_bash.py \
    --deepspeed ds_z3_config.json \
    --stage ${stage1} \
    --do_train True \
    --model_name_or_path ${model_name_or_path} \
    --dataset ${dataset} \
    --dataset_dir ../../data \
    --template default \
    --finetuning_type ${finetuning_type} \
    --output_dir ${full_sft_output_dir} \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1600 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2  \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps ${step} \
    --learning_rate ${lr} \
    --num_train_epochs 40.0 \
    --max_samples 100000 \
    --ddp_timeout 1800000 \
    --optim adamw_torch \
    --packing True \
    --plot_loss \
    --bf16 True \
    --val_size ${val_size} \
    --dataset_eval 'eval_all' \
    --eval_steps ${step} \
    --evaluation_strategy steps
# lora predict
cp full.sh $full_sft_output_dir
find $full_sft_output_dir -type d -name "global_step*" -exec rm -rf {} +
cd MY_PATH/comp/my_finetune/LLaMA-Factory

exit

dataset='eval_all'
finetuning_type='full'
model_name_or_path=$full_sft_output_dir
CUDA_VISIBLE_DEVICES=$eval_device python src/train_bash.py \
    --stage sft \
    --model_name_or_path ${full_sft_output_dir} \
    --finetuning_type ${finetuning_type} \
    --template default \
    --dataset_dir data \
    --dataset ${dataset} \
    --cutoff_len 1925 \
    --max_samples 100000 \
    --per_device_eval_batch_size 8 \
    --predict_with_generate True \
    --max_new_tokens 256 \
    --top_p 0.5 \
    --bf16 True \
    --temperature 0.5 \
    --output_dir ${full_sft_output_dir} \
    --do_predict True 

# transform format and calculate score
cd MY_PATH/comp/my_finetune/LLaMA-Factory/examples/full_multi_gpu/llm_data
# 词嵌入模型地址
export MODEL_PATH="MY_PATH/comp/used_data/GoogleNews-vectors-negative300.bin"

# 原始submission文件地址
original_submission_path="MY_PATH/comp/my_finetune/LLaMA-Factory/examples/full_multi_gpu/llm_data/submission.jsonl"
# factory生成的预测文件地址
factory_pred_path="$full_sft_output_dir/generated_predictions.jsonl"
# 生成的submission文件地址
submit_jsonl_path="$(dirname "$factory_pred_path")/submission.jsonl"
# 答案文件地址
answer_jsonl_path="MY_PATH/comp/my_finetune/LLaMA-Factory/examples/full_multi_gpu/llm_data/ref_data.jsonl"
refined_answer_jsonl_path="MY_PATH/comp/my_finetune/LLaMA-Factory/examples/full_multi_gpu/llm_data/ref_data_refined.jsonl"
# 分数报告地址
score_report_path="$(dirname "$factory_pred_path")/score.txt"
refined_score_report_path="$(dirname "$factory_pred_path")/refined_score.txt"

red="\e[31m"
reset="\e[0m"

#  生成submission文件
echo -e "${red}Generating submission file...${reset}"
echo -e "${red}Submission file created at: \n$submit_jsonl_path${reset}"
python3 generate_submission_from_factory.py --original_submission_path $original_submission_path --factory_pred_path $factory_pred_path --generated_submission_path $submit_jsonl_path

# 评分
echo -e "${red}Evaluating...${reset}"
echo -e "${red}Score report created at: \n$score_report_path${reset}"
python3 evaluate.py --answer_jsonl_path $answer_jsonl_path --submit_jsonl_path $submit_jsonl_path > $score_report_path
echo -e "${red}Refined core report created at: \n$refined_score_report_path${reset}"
python3 evaluate_refined.py --answer_jsonl_path $refined_answer_jsonl_path --submit_jsonl_path $submit_jsonl_path > $refined_score_report_path