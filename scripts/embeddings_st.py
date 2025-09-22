import os
import glob
import sys
import subprocess

import dotenv
dotenv.load_dotenv()
from sentence_transformers import SentenceTransformer
import pandas as pd

DATA_PATH = './data'
EMBEDDINGS_DIR = './data_embeddings'

DEFAULT_MODEL_ID = 'all-MiniLM-L6-v2'

def get_file_embeddings(path: str, model: SentenceTransformer, batch_size: int = 32) -> pd.DataFrame:
    data = pd.read_csv(path, sep=',', index_col=0)
    print(data.head())
    sentences = data['sentences'].tolist()

    embeddings = pd.DataFrame(model.encode(sentences, batch_size=batch_size))

    print("Memory after embeddings extraction:")
    print(subprocess.run(['nvidia-smi']))

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

    if len(sys.argv) > 3 and sys.argv[3].isdigit():
        batch_size = int(sys.argv[3])
    else:
        batch_size = 32

    model = SentenceTransformer(model_id, model_kwargs={'device_map': 'auto'})
    print("Model Device:", model.device)
    print("Memory after model load:")
    print(subprocess.run(['nvidia-smi']))

    destination_dir = os.path.join(EMBEDDINGS_DIR, model_id)
    os.makedirs(destination_dir, exist_ok=True)

    if filename == '':
        for file in glob.glob(os.path.join(DATA_PATH, '*.csv')):
            embeddings = get_file_embeddings(file, model, batch_size)
            name = os.path.split(file)[1]
            embeddings.to_csv(os.path.join(destination_dir, name), sep=',')
    else:
        embeddings = get_file_embeddings(os.path.join(DATA_PATH, filename), model, batch_size)
        embeddings.to_csv(os.path.join(destination_dir, filename), sep=',')

if __name__ == '__main__':
    main()
