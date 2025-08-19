import os
import glob
import sys

import dotenv
dotenv.load_dotenv()
from sentence_transformers import SentenceTransformer
import pandas as pd

DATA_PATH = './data'
EMBEDDINGS_DIR = './data_embeddings'

DEFAULT_MODEL_ID = 'all-MiniLM-L6-v2'

def get_file_embeddings(path: str, model: SentenceTransformer) -> pd.DataFrame:
        data = pd.read_csv(path, sep=',')
        print(data.head())
        sentences = data['sentences'].tolist()

        embeddings = pd.DataFrame(model.encode(sentences))
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

    model = SentenceTransformer(model_id)

    destination_dir = os.path.join(EMBEDDINGS_DIR, model_id)
    os.makedirs(destination_dir, exist_ok=True)

    if filename == '':
        for file in glob.glob(os.path.join(DATA_PATH, '*.csv')):
            embeddings = get_file_embeddings(file, model)
            name = os.path.split(file)[1]
            embeddings.to_csv(os.path.join(destination_dir, name), sep=',')
    else:
        embeddings = get_file_embeddings(os.path.join(DATA_PATH, filename), model)
        embeddings.to_csv(os.path.join(destination_dir, filename), sep=',')

if __name__ == '__main__':
    main()
