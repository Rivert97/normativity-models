# Module imports
import os
import sys

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import pandas as pd

# Global variables
DATASET_NAME = 'Rivert97/ug-normativity'
EMBEDDINGS_DIR = './dataset_embeddings'

DEFAULT_MODEL_ID = 'all-MiniLM-L6-v2'

def get_embeddings(dataset, document_name: str, model: SentenceTransformer) -> pd.DataFrame:
    filtered = dataset.filter(lambda row: row['title'] == document_name)
    embeddings = model.encode(filtered['question'])

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

    # Loading embeddings model
    model = SentenceTransformer(model_id)

    # Loading the questions
    dataset = load_dataset(DATASET_NAME)
    dataset = dataset['train']
    print(dataset)

    # Converting questions to embeddings
    destination_dir = os.path.join(EMBEDDINGS_DIR, model_id)
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
