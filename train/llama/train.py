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

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
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

def _tokenize_fn(
    texts: Sequence[str],
    tokenizer: LlamaTokenizer,
    ) -> Dict:
    """
    对文本序列进行tokenize

    Args:
        texts (Sequence[str]): 文本序列
        tokenizer (LlamaTokenizer)

    Returns:
        Dict: input_ids, labels, input_ids_lens, labels_lens
    """
    tokenized_list = [
        tokenizer(
            text=text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation = True,
        )
        for text in texts
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    # 每个序列的真实长度
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )
    
def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: LlamaTokenizer,
    ) -> Dict:
    """
    使用tokenizer对文本进行tokenize

    Args:
        sources (Sequence[str]): 输入
        targets (Sequence[str]): 输出
        tokenizer (LlamaTokenizer): llama的tokenizer
    """
    examples = [s + t for s, t in zip(sources, targets)]
    
    examples_tokenized = _tokenize_fn(examples, tokenizer)
    sources_tokenized = _tokenize_fn(sources, tokenizer)
    
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    
    return dict(input_ids=input_ids, labels=labels)
    
    
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
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]
        
        logging.warning("Tokenizing inputs...")
        data_dict = preprocess(sources, targets, tokenizer)
    
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        return dict(inpu_ids=self.input_ids[index], labels=self.labels[index])
        
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
        model_max_len=training_args.max_seq_len,
        padding_side="right",
        use_fast=False,
    )
    
    
    
if __name__ == '__main__':
    train()