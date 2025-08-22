import os
import glob
import sys

import dotenv
dotenv.load_dotenv()
import pandas as pd

from llama_cpp import Llama

DATA_PATH = './data'
EMBEDDINGS_DIR = './data_embeddings'

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

def get_file_embeddings(path: str, model) -> pd.DataFrame:
    data = pd.read_csv(path, sep=',')
    print(data.head())
    sentences = data['sentences'].tolist()

    all_embeddings = []

    for sentence in sentences:
        all_embeddings.append(model.embed(sentence))

    embeddings = pd.DataFrame(all_embeddings)
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

    destination_dir = os.path.join(EMBEDDINGS_DIR, model_gguf)
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
