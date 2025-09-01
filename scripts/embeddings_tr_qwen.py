import os
import glob
import sys
import subprocess

import dotenv
dotenv.load_dotenv()
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np

DATA_PATH = './data'
EMBEDDINGS_DIR = './data_embeddings_transformers'

DEFAULT_MODEL_ID = 'Qwen/Qwen3-Embedding-0.6B'

def get_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = AutoModel.from_pretrained(model_name, device_map='auto')

    print("Memory after model load:")
    print(subprocess.run(['nvidia-smi']))

    return model, tokenizer

def get_file_embeddings(path: str, model, tokenizer, batch_size = 32) -> pd.DataFrame:
    data = pd.read_csv(path, sep=',')
    print(data.head())
    sentences = data['sentences'].tolist()

    max_length = 8192

    all_embeddings = []

    for start in range(0, len(sentences), batch_size):
        batch = sentences[start:start+batch_size]
        # Tokenize the input texts
        batch_dict = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**batch_dict)
        sentence_embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        all_embeddings.append(sentence_embeddings.cpu().numpy())

    print("Memory after embeddings extraction:")
    print(subprocess.run(['nvidia-smi']))

    embeddings = pd.DataFrame(np.vstack(all_embeddings))
    print(embeddings.head())

    return embeddings

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def main():
    if len(sys.argv) > 1:
        model_id = sys.argv[1]
    else:
        model_id = DEFAULT_MODEL_ID

    if len(sys.argv) > 2:
        filename = sys.argv[2]
    else:
        filename = '' # Process al files

    if len(sys.argv) > 3 and sys.argv[3].isdigit():
        batch_size = int(sys.argv[3])
    else:
        batch_size = 32

    model, tokenizer = get_model(model_id)

    destination_dir = os.path.join(EMBEDDINGS_DIR, model_id)
    os.makedirs(destination_dir, exist_ok=True)

    if filename == '':
        for file in glob.glob(os.path.join(DATA_PATH, '*.csv')):
            embeddings = get_file_embeddings(file, model, tokenizer, batch_size)
            name = os.path.split(file)[1]
            embeddings.to_csv(os.path.join(destination_dir, name), sep=',')
    else:
        embeddings = get_file_embeddings(os.path.join(DATA_PATH, filename), model, tokenizer, batch_size)
        embeddings.to_csv(os.path.join(destination_dir, filename), sep=',')

if __name__ == '__main__':
    main()
