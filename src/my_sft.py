# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/summarization/run_summarization.py

from typing import TYPE_CHECKING, List, Optional

from transformers import DataCollatorForSeq2Seq
IGNORE_INDEX = -100
from ...model import load_model, load_tokenizer
from ...train.sft.metric import ComputeMetrics
from ...train.sft.trainer import CustomSeq2SeqTrainer
from ...train.utils import create_modelcard_and_push
from ..utils import create_custom_optimzer
from transformers import InfNanRemoveLogitsProcessor, LogitsProcessorList, PreTrainedModel
from transformers.utils import (
    SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
    is_torch_bf16_gpu_available,
    is_torch_cuda_available,
    is_torch_mps_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)
from transformers import Seq2SeqTrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.versions import require_version
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Sequence, Tuple, Union

import numpy as np

@dataclass
class ComputeMetrics:
    r"""
    Wraps the tokenizer into metric functions, used in Seq2SeqPeftTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def __call__(self, eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]]) -> Dict[str, float]:
        r"""
        Uses the model predictions to compute metrics.
        """
        preds, labels = eval_preds
        score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        return {k: float(np.mean(v)) for k, v in score_dict.items()}

def get_logits_processor() -> "LogitsProcessorList":
    r"""
    Gets logits processor that removes NaN and Inf logits.
    """
    logits_processor = LogitsProcessorList()
    logits_processor.append(InfNanRemoveLogitsProcessor())
    return logits_processor


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments
def split_dataset(
    dataset: Union["Dataset", "IterableDataset"], data_args: "DataArguments", training_args: "Seq2SeqTrainingArguments"
) -> Dict[str, "Dataset"]:
    if training_args.do_train:
        if data_args.val_size > 1e-6:  # Split the dataset
            if data_args.streaming:
                val_set = dataset.take(int(data_args.val_size))
                train_set = dataset.skip(int(data_args.val_size))
                dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)
                return {"train_dataset": train_set, "eval_dataset": val_set}
            else:
                val_size = int(data_args.val_size) if data_args.val_size > 1 else data_args.val_size
                dataset = dataset.train_test_split(test_size=val_size, seed=training_args.seed)
                return {"train_dataset": dataset["train"], "eval_dataset": dataset["test"]}
        else:
            if data_args.streaming:
                dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)
            return {"train_dataset": dataset}
    else:  # do_eval or do_predict
        return {"eval_dataset": dataset}

def get_dataset(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo"],
    # split: Optional[str] = "train", # TODO: add split
) -> Union["Dataset", "IterableDataset"]:
    template = get_template_and_fix_tokenizer(tokenizer, data_args.template)
    if data_args.train_on_prompt and template.efficient_eos:
        raise ValueError("Current template does not support `train_on_prompt`.")
    # Load from cache
    if data_args.cache_path is not None:
        if os.path.exists(data_args.cache_path):
            logger.warning("Loading dataset from disk will ignore other data arguments.")
            dataset = load_from_disk(data_args.cache_path)
            if data_args.streaming:
                dataset = dataset.to_iterable_dataset()
            return dataset

        if data_args.streaming:
            raise ValueError("Turn off `streaming` when saving dataset to disk.")

    with training_args.main_process_first(desc="load dataset"):
        all_datasets = []
        for dataset_attr in get_dataset_list(data_args):
            all_datasets.append(load_single_dataset(dataset_attr, model_args, data_args))
        dataset = merge_dataset(all_datasets, data_args, training_args)

    with training_args.main_process_first(desc="pre-process dataset"):
        preprocess_func, print_function = get_preprocess_and_print_func(
            tokenizer, template, data_args, training_args, stage
        )
        column_names = list(next(iter(dataset)).keys())
        kwargs = {}
        if not data_args.streaming:
            kwargs = dict(
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=(not data_args.overwrite_cache),
                desc="Running tokenizer on dataset",
            )

        dataset = dataset.map(preprocess_func, batched=True, remove_columns=column_names, **kwargs)

        if data_args.cache_path is not None and not os.path.exists(data_args.cache_path):
            if training_args.should_save:
                dataset.save_to_disk(data_args.cache_path)
                logger.info("Dataset cache saved at {}.".format(data_args.cache_path))

        if training_args.should_log:
            try:
                print_function(next(iter(dataset)))
            except StopIteration:
                raise RuntimeError("Cannot find valid samples, check `data/README.md` for the data format.")

        return dataset


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer = load_tokenizer(model_args)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    # Customize dataset loading behavior
    dataset_split = {}
    if data_args.dataset_eval:
        # copy data_args and ignore val_size param
        import copy
        customized_data_args = copy.copy(data_args)
        customized_data_args.val_size = 0.0
        # dataset-train
        dataset = get_dataset(tokenizer, model_args, customized_data_args, training_args, stage="sft")
        train_set = split_dataset(dataset, customized_data_args, training_args)["train_dataset"]
        # dataset-eval
        customized_data_args.dataset = customized_data_args.dataset_eval
        dataset_eval = get_dataset(tokenizer, model_args, customized_data_args, training_args, stage="sft")
        # here, "train_dataset" is actually "eval_dataset"
        eval_set = split_dataset(dataset_eval, customized_data_args, training_args)["train_dataset"]
        # dict
        dataset_split = {"train_dataset": train_set, "eval_dataset": eval_set}
    else:
        dataset = get_dataset(tokenizer, model_args, data_args, training_args, stage="sft")
        dataset_split = split_dataset(dataset, data_args, training_args)

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if tokenizer.padding_side == "right" else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams

    # Initialize our Trainer
    optimizer = create_custom_optimzer(model, dataset, training_args, finetuning_args)
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        optimizers=(optimizer, None),
        compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
        **dataset_split,
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
    # Evaluation
    if training_args.do_eval and not training_args.do_train:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if training_args.predict_with_generate:  # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
   
        predict_results = trainer.predict(dataset, metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate:  # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)