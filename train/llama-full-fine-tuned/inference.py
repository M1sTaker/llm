import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

model_path = "Llama-2-7b-hf"
print("loading model:", model_path)

model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto", low_cpu_mem_usage=True)
tokenizer = LlamaTokenizer.from_pretrained(model_path)
print("Human:")
line = input()
while line:
    inputs = "Human: " + line.strip() + "\n\nAssistant:"
    input_ids = tokenizer(inputs, return_tensors="pt").input_ids
    outputs = model.generate(input_ids,
                             max_new_tokens=100,
                             do_sample=True,
                             top_k=30,
                             top_p=0.85,
                             temperature = 0.5,
                             repetition_penalty=1.,
                             eos_token_id=2,
                             bos_token_id=1,
                             pad_token_id=0
                            )
    res = tokenizer.batch_decode(outputs,
                                 skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)
    print("Assistant:\n" + res[0].strip().replace(inputs, ""))
    print("\n----------------------------------\nHuman:")
    line = input()