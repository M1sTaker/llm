import torch
from torch.utils.data import Dataset
import transformers
from transformers import (
    HfArgumentParser,
    LlamaForCausalLM,
    LlamaTokenizer
)

from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import logging
import copy
import json

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

@dataclass
class ModelArguments:
    """
    模型参数
    """
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf")

@dataclass
class DataArguments:
    """
    数据参数
    """
    data_path: Optional[str] = field(default=None, metadata={"help": "path to training data."})
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """
    训练参数
    """
    # 缓存目录，存放预训练checkpoint
    cache_dir: Optional[str] = field(default=None)
    max_seq_len: int = field(
        default=512,
        metadata={"help": "Maximum sequence length."}
    )
    
class SupervisedDataset(Dataset):
    """
    使用alpaca-lora数据集作为有监督的数据集 将aplaca-lora中的数据嵌入到prompt中
    使用tokenizer对文本进行预处理
    """
    
    def __init__(self, data_path: str, tokenizer: LlamaTokenizer) -> None:
        super().__init__()
        logging.warning("Loading data...")
        
        with open(data_path, 'r') as f:
            list_data_dict = json.load(f)
        
        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example["output"]}{tokenizer.eos_token}" for example in list_data_dict]
        
        logging.warning("Tokenizing inputs...")
        data_dict = preprocess(sources, targets, tokenizer)
        
def prepare_supervised_data_module(tokenizer: LlamaTokenizer, data_args) -> Dict:
    """
    生成有监督的数据集和collator
    
    Returns:
        Dict: train_dataset, eval_dataset, data_collator
    """
    

def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    
    tokenizer = LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        max_seq_len=training_args.max_seq_len,
        padding_side="right",
        use_fast=False,
    )
    
    
    
if __name__ == '__main__':
    train()