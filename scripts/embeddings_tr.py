import os
import glob
import sys

import dotenv
dotenv.load_dotenv()
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch

DATA_PATH = './data'
EMBEDDINGS_DIR = './data_embeddings'

DEFAULT_MODEL_ID = 'bert-base-uncased'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    return model, tokenizer

def get_file_embeddings(path: str, model, tokenizer) -> pd.DataFrame:
    data = pd.read_csv(path, sep=',')
    print(data.head())
    sentences = data['sentences'].tolist()

    inputs = tokenizer(sentences,
                       truncation=True,
                       padding=True,
                       return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Token-level embeddings
    attention_mask = inputs['attention_mask']
    last_hidden = outputs.last_hidden_state

    # Mean pooling
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    masked_hidden = last_hidden * mask

    sum_hidden = masked_hidden.sum(dim=1)
    sum_mask = mask.sum(dim=1).clamp(min=1e-9)

    sentence_embeddings = sum_hidden / sum_mask

    embeddings = pd.DataFrame(sentence_embeddings.cpu())
    print(embeddings.head())

    return embeddings

def main():
    if len(sys.argv) > 1:
        model_id = sys.argv[1]
    else:
        model_id = DEFAULT_MODEL_ID

    if len(sys.argv) > 2:
        filename = sys.argv[2]
    else:
        filename = '' # Process al files

    model, tokenizer = get_model(model_id)

    destination_dir = os.path.join(EMBEDDINGS_DIR, model_id)
    os.makedirs(destination_dir, exist_ok=True)

    if filename == '':
        for file in glob.glob(os.path.join(DATA_PATH, '*.csv')):
            embeddings = get_file_embeddings(file, model, tokenizer)
            name = os.path.split(file)[1]
            embeddings.to_csv(os.path.join(destination_dir, name), sep=',')
    else:
        embeddings = get_file_embeddings(os.path.join(DATA_PATH, filename), model, tokenizer)
        embeddings.to_csv(os.path.join(destination_dir, filename), sep=',')

if __name__ == '__main__':
    main()
