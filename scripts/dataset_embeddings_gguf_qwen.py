# Module imports
import os
import sys

import dotenv
dotenv.load_dotenv()
import pandas as pd
from datasets import load_dataset
import numpy as np

from llama_cpp import Llama

# Global variables
MODEL_DIR = '/home/rgarcia/tesis/00_src/run/models/hub/models--JonathanMiddleton--Qwen3-Embedding-8B-GGUF/snapshots/875e1df98b106c2471e0bc8fd41121f425f789cf'
DATASET_NAME = 'Rivert97/ug-normativity'
EMBEDDINGS_DIR = './dataset_embeddings_transformers/Qwen/'

DEFAULT_MODEL_GGUF = 'Qwen3-Embedding-8B-Q4_K_M'

def get_model(model_path: str):
    model = Llama(
        model_path=model_path,
        embedding=True,
        n_ctx=2048,
        n_gpu_layers=-1,
    )

    return model

def get_embeddings(dataset, document_name: str, model, batch_size = 32) -> pd.DataFrame:
    import pdb; pdb.set_trace()
    filtered = dataset.filter(lambda row: row['title'] == document_name)
    questions = filtered['question']

    all_embeddings = []

    for start in range(0, len(questions), batch_size):
        batch = questions[start:start+batch_size]
        all_embeddings.append(model.embed(batch).cpu())

    embeddings = pd.DataFrame(np.vstack(all_embeddings))
    embeddings.index = filtered['id']
    print(embeddings.head())

    return embeddings

def main():
    if len(sys.argv) > 1:
        model_id = sys.argv[1]
    else:
        model_id = DEFAULT_MODEL_GGUF

    if len(sys.argv) > 2:
        filename = sys.argv[2]
    else:
        filename = '' # Process al files

    if len(sys.argv) > 3 and sys.argv[3].isdigit():
        batch_size = int(sys.argv[3])
    else:
        batch_size = 32

    # Loading embeddings model
    model, tokenizer = get_model(os.path.join(MODEL_DIR, f"{model_id}.gguf"))

    # Loading the questions
    dataset = load_dataset(DATASET_NAME)
    dataset = dataset['train']
    print(dataset)

    # Converting questions to embeddings
    destination_dir = os.path.join(EMBEDDINGS_DIR, model_id)
    os.makedirs(destination_dir, exist_ok=True)
    if filename == '':
        for title in sorted(set(dataset['title'])):
            embeddings = get_embeddings(dataset, title, model, batch_size)
            embeddings.to_csv(os.path.join(destination_dir, f"{title}.csv"), sep=',')
    else:
        embeddings = get_embeddings(dataset, filename, model, batch_size)
        embeddings.to_csv(os.path.join(destination_dir, f"{filename}.csv"), sep=',')

if __name__ == '__main__':
    main()
