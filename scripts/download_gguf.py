import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

if len(sys.argv) < 3:
    print("Please specify a model and a gguf file")
    sys.exit(1)

model_id = sys.argv[1]
filename = sys.argv[2]

torch_dtype = torch.float32 # could be torch.float16 or torch.bfloat16 too
tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)
model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=filename, torch_dtype=torch_dtype)
print("Device", model.device)