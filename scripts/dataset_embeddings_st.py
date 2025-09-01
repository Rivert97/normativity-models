# Module imports
import os
import sys
import subprocess

import dotenv
dotenv.load_dotenv()
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Global variables
DATASET_NAME = 'Rivert97/ug-normativity'
EMBEDDINGS_DIR = './dataset_embeddings'

DEFAULT_MODEL_ID = 'all-MiniLM-L6-v2'

def get_embeddings(dataset, document_name: str, model: SentenceTransformer, batch_size: int = 32) -> pd.DataFrame:
    filtered = dataset.filter(lambda row: row['title'] == document_name)
    embeddings = model.encode(filtered['question'], batch_size=batch_size)

    print("Memory after embeddings extraction:")
    print(subprocess.run(['nvidia-smi']))

    df = pd.DataFrame(embeddings)
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

    if len(sys.argv) > 3 and sys.argv[3].isdigit():
        batch_size = int(sys.argv[3])
    else:
        batch_size = 32

    # Loading embeddings model
    model = SentenceTransformer(model_id, model_kwargs={'device_map': 'auto'})
    print("Model Device:", model.device)
    print("Memory after model load:")
    print(subprocess.run(['nvidia-smi']))

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
