import torch
from torch.utils.data import Dataset
import transformers
from transformers import (
    HfArgumentParser,
    LlamaForCausalLM,
    LlamaTokenizer,
    Trainer
)

from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import logging
import copy
import json

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
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

def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: LlamaTokenizer,
    model: LlamaForCausalLM
):
    """
    由于llama在预训练时没有pad token，在微调时添加pad token需要对tokenizer和embedding进行resize
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg        

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
    
    # 将target之前的文本source去除
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

@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: LlamaTokenizer
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        )
        
def prepare_supervised_data_module(tokenizer: LlamaTokenizer, data_args) -> Dict:
    """
    生成有监督的数据集和collator
    
    Returns:
        Dict: train_dataset, eval_dataset, data_collator
    """
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
    

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
        model_max_length=training_args.max_seq_len,
        padding_side="right",
        use_fast=False,
    )
    
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    
    # tokenizer.add_special_tokens(
    #     {
    #         "eos_token": DEFAULT_EOS_TOKEN,
    #         "bos_token": DEFAULT_BOS_TOKEN,
    #         "unk_token": DEFAULT_UNK_TOKEN,
    #     }
    # )
    
    data_module = prepare_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    
if __name__ == '__main__':
    train()