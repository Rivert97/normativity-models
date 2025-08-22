# Module imports
import os
import sys

import dotenv
dotenv.load_dotenv()
import pandas as pd
from datasets import load_dataset

from llama_cpp import Llama

# Global variables
DATASET_NAME = 'Rivert97/ug-normativity'
EMBEDDINGS_DIR = './dataset_embeddings/'

DEFAULT_MODEL_DIR = '$HOME/cache/huggingface/hub/models--ChristianAzinn--gist-embedding-v0-gguf/snapshots/4a2a322d1f8c2bd0438157958f4f8f516a63cf22'
DEFAULT_MODEL_GGUF = 'gist-embedding-v0.Q4_K_M'

def get_model(model_path: str):
    model = Llama(
        model_path=model_path,
        embedding=True,
        n_ctx=2048,
        n_gpu_layers=-1,
    )

    return model

def get_embeddings(dataset, document_name: str, model) -> pd.DataFrame:
    filtered = dataset.filter(lambda row: row['title'] == document_name)
    questions = filtered['question']

    all_embeddings = []

    for question in questions:
        all_embeddings.append(model.embed(question))

    embeddings = pd.DataFrame(all_embeddings)
    embeddings.index = filtered['id']
    print(embeddings.head())

    return embeddings

def main():
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    else:
        model_dir = DEFAULT_MODEL_DIR

    if len(sys.argv) > 2:
        model_gguf = sys.argv[2]
    else:
        model_gguf = DEFAULT_MODEL_GGUF

    if len(sys.argv) > 3:
        filename = sys.argv[3]
    else:
        filename = '' # Process al files

    # Loading embeddings model
    model = get_model(os.path.join(model_dir, f"{model_gguf}.gguf"))

    # Loading the questions
    dataset = load_dataset(DATASET_NAME)
    dataset = dataset['train']
    print(dataset)

    # Converting questions to embeddings
    destination_dir = os.path.join(EMBEDDINGS_DIR, model_gguf)
    os.makedirs(destination_dir, exist_ok=True)
    if filename == '':
        for title in sorted(set(dataset['title'])):
            embeddings = get_embeddings(dataset, title, model)
            embeddings.to_csv(os.path.join(destination_dir, f"{title}.csv"), sep=',')
    else:
        embeddings = get_embeddings(dataset, filename, model)
        embeddings.to_csv(os.path.join(destination_dir, f"{filename}.csv"), sep=',')

if __name__ == '__main__':
    main()
