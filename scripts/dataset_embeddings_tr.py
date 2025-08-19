# Module imports
import os
import sys

import dotenv
dotenv.load_dotenv()
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch

# Global variables
DATASET_NAME = 'Rivert97/ug-normativity'
EMBEDDINGS_DIR = './dataset_embeddings'

DEFAULT_MODEL_ID = 'bert-base-uncased'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    return model, tokenizer

def get_embeddings(dataset, document_name: str, model, tokenizer) -> pd.DataFrame:
    filtered = dataset.filter(lambda row: row['title'] == document_name)
    questions = filtered['question']

    inputs = tokenizer(questions,
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

    df = pd.DataFrame(sentence_embeddings.cpu())
    df.index = filtered['id']
    print(df.head())

    return df

def main():
    if len(sys.argv) > 1:
        model_id = sys.argv[1]
    else:
        model_id = DEFAULT_MODEL_ID

    if len(sys.argv) > 2:
        filename = sys.argv[2]
    else:
        filename = '' # Process al files

    # Loading embeddings model
    model, tokenizer = get_model(model_id)

    # Loading the questions
    dataset = load_dataset(DATASET_NAME)
    dataset = dataset['train']
    print(dataset)

    # Converting questions to embeddings
    destination_dir = os.path.join(EMBEDDINGS_DIR, model_id)
    os.makedirs(destination_dir, exist_ok=True)
    if filename == '':
        for title in sorted(set(dataset['title'])):
            embeddings = get_embeddings(dataset, title, model, tokenizer)
            embeddings.to_csv(os.path.join(destination_dir, f"{title}.csv"), sep=',')
    else:
        embeddings = get_embeddings(dataset, filename, model, tokenizer)
        embeddings.to_csv(os.path.join(destination_dir, f"{filename}.csv"), sep=',')

if __name__ == '__main__':
    main()
