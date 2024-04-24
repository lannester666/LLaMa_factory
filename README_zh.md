# 介绍

原仓库地址可见[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).本次比赛的所有训练脚本可见examples/full_multi_gpu,所用数据集可见data/。复现训练结果需要首先下载对应的Base模型。部分模型下载地址：


BioMistral7B：
https://huggingface.co/BioMistral/BioMistral-7B

QWen1.5-7B-Chat:
https://huggingface.co/Qwen/Qwen1.5-7B-Chat

Mistral7B-base:
https://huggingface.co/mistralai/Mistral-7B-v0.1

Mistral7B-Instruct：
https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2



**easy start**

```
cd examples/full_multi_gpu
bash full.sh
```
注意需要将脚本内的路径更换为自己的路径。