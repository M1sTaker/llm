from transformers import AutoModel, AutoTokenizer
import torch
from peft import PeftModel
import json

model = AutoModel.from_pretrained("chatglm-6b",
                                  trust_remote_code=True,
                                  device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("chatglm-6b",
                                          trust_remote_code=True)

model = PeftModel.from_pretrained(model, "chatglm-6b-lora")

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