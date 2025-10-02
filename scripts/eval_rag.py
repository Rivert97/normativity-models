import os
import glob
import subprocess

import dotenv
dotenv.load_dotenv()
import pandas as pd
import numpy as np
from sentence_transformers import util
import torch
from rouge_score import rouge_scorer, scoring
from datasets import load_dataset

from .lib.data import Document
from .lib.models import ModelBuilder

# Gobal variables
MODELS = [
    {'embeddings_id': 'Qwen/Qwen3-Embedding-8B', 'model_id': 'Qwen/Qwen3-0.6B'},
    {'embeddings_id': 'Qwen/Qwen3-Embedding-8B', 'model_gguf': '/home/rgarcia/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-GGUF/snapshots/23749fefcc72300e3a2ad315e1317431b06b590a/Qwen3-0.6B-Q8_0.gguf'},
]
DATA_PATH = './data'
DATA_EMBEDDINGS_PATH = f'./data_embeddings'
DATASET_EMBEDDINGS_PATH = f'./dataset_embeddings'
DATASET_NAME = 'Rivert97/ug-normativity'
RESULTS_DIR = './results_rag'
RESPONSES_DIR = './responses'
START = 0
END = None

# Other variables
k = 5
file_excludes = []
score_metric = 'dot' # dot, cos

def load_csv_data(data_embeddings_dirname, dataset_embeddings_dirname):
    all_data = []
    all_embeddings = []
    all_questions_embeddings = []

    # Loading the data
    for data_filename in glob.glob(os.path.join(DATA_PATH, '*.csv')):
        embeddings_filename = os.path.join(data_embeddings_dirname, os.path.split(data_filename)[1])
        questions_filename = os.path.join(dataset_embeddings_dirname, os.path.split(data_filename)[1])

        if not os.path.exists(questions_filename):
            print(f"Ignoring file {data_filename}")
            file_excludes.append(os.path.splitext(os.path.split(data_filename)[1])[0])
            continue

        if os.path.splitext(os.path.split(data_filename)[1])[0] in file_excludes:
            continue

        data = pd.read_csv(data_filename, sep=',', index_col=0)
        embeddings = pd.read_csv(embeddings_filename, sep=',', index_col=0)
        questions_embeddings = pd.read_csv(questions_filename, sep=',', index_col=0)

        all_data.append(data)
        all_embeddings.append(embeddings)
        all_questions_embeddings.append(questions_embeddings)

    data = pd.concat(all_data, ignore_index=True)
    embeddings = pd.concat(all_embeddings, ignore_index=True)
    questions_embeddings = pd.concat(all_questions_embeddings)

    return data, embeddings, questions_embeddings

def find_questions_related_chunks(dataset, questions_embeddings, data, embeddings):
    # For each question find its chunk
    questions = []

    for question in dataset:
        if question['title'] in file_excludes:
            continue

        tmp_question = {
            'question': question,
            'question_embeddings': questions_embeddings.loc[question['id']].values,
            'chunk_idx': [],
            'chunk': [],
            'chunk_embeddings': [],
        }

        for path, chunks in data.loc[data['document_name'] == question['title']].groupby('path'):
            if path.lower().endswith(question['context'].lower().strip()):
                for chunk_idx, chunk in chunks.sort_values('num').iterrows():
                    tmp_question['chunk_idx'].append(chunk_idx)
                    tmp_question['chunk'].append(chunk)
                    tmp_question['chunk_embeddings'].append(embeddings.loc[chunk_idx].values)

        if tmp_question['chunk_idx'] is not None:
            questions.append(tmp_question)

    return questions

def get_top_k_scores_info(questions, data, embeddings):
    # Analyzing score from each question to its K neighbors
    top_k_info = {
        'scores': [],
        'scores_mean': [],
        'embeddings_idx': {},
    }
    for question in questions:
        if score_metric == 'cos':
            scores = 1.0 - np.dot(question['question_embeddings'], embeddings.values.T)/(np.linalg.norm(question['question_embeddings'])*np.linalg.norm(embeddings.values, axis=1))
            reversed_sort = False
        else:
            scores = util.dot_score(question['question_embeddings'], embeddings.values)[0].tolist()
            reversed_sort = True
        doc_score_pairs = list(zip(data.index.tolist(), scores))

        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=reversed_sort)

        dist = [d[1] for d in doc_score_pairs[:k]]
        idx = [d[0] for d in doc_score_pairs[:k]]

        top_k_info['scores'].extend(dist)
        top_k_info['scores_mean'].append(np.mean(dist))
        top_k_info['embeddings_idx'][question['question']['id']] = idx

    return top_k_info

def calculate_rouge(predicted, reference):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    for pred, ref in zip(predicted, reference):
        score = scorer.score(ref, pred)
        aggregator.add_scores(score)
    result = aggregator.aggregate()

    return {metric: result[metric].mid.fmeasure for metric in result}

def calculate_rouge_by_file(predicted, reference, model_opts, model_name):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    files = predicted['file'].unique()
    metrics = {
        'rouge1': np.zeros((len(files),), dtype=np.float32),
        'rouge2': np.zeros((len(files),), dtype=np.float32),
        'rougeL': np.zeros((len(files),), dtype=np.float32),
        'rougeLsum': np.zeros((len(files),), dtype=np.float32),
    }
    for f_idx, f in enumerate(files):
        file_predicted = predicted[predicted['file'] == f]
        file_reference = reference[reference['file'] == f]

        for pred, ref in zip(file_predicted['text'], file_reference['text']):
            single_score = scorer.score(ref, pred)
            aggregator.add_scores(single_score)
        result = aggregator.aggregate()
        score = {metric: result[metric].mid.fmeasure for metric in result}

        metrics['rouge1'][f_idx] = score['rouge1']
        metrics['rouge2'][f_idx] = score['rouge2']
        metrics['rougeL'][f_idx] = score['rougeL']
        metrics['rougeLsum'][f_idx] = score['rougeLsum']

    model_identifier = f"{model_opts['embeddings_id']}_{model_name}"
    return pd.DataFrame({
            f'{model_identifier} rouge1': metrics['rouge1'],
            f'{model_identifier} rouge2': metrics['rouge2'],
            f'{model_identifier} rougeL': metrics['rougeL'],
            f'{model_identifier} rougeLsum': metrics['rougeLsum']
        },
        index=files)

def update_csv_data(filename, new_data, axis=0):
    if os.path.exists(filename):
        prev_df = pd.read_csv(filename, sep=',', index_col=0)
        if axis == 0:
            if set(new_data.index).intersection(set(prev_df.index)):
                for idx in new_data.index:
                    prev_df.loc[idx] = new_data.loc[idx]
                updated_df = prev_df
            else:
                updated_df = pd.concat([prev_df, new_data], ignore_index=False, axis=axis)
        else:
            if set(new_data.columns).intersection(set(prev_df.columns)):
                for col in new_data.columns:
                    prev_df[col] = new_data[col]
                updated_df = prev_df
            else:
                updated_df = pd.concat([prev_df, new_data], ignore_index=False, axis=axis)
    else:
        updated_df = new_data

    updated_df.to_csv(filename, sep=',')

for model_opts in MODELS:
    if 'model_id' in model_opts:
        print(f"Procesando {model_opts['embeddings_id']} with {model_opts['model_id']}")
        model_name = model_opts['model_id']
    elif 'model_gguf' in model_opts:
        print(f"Procesando {model_opts['embeddings_id']} with {model_opts['model_gguf'].split('/')[-1]}")
        model_name = os.path.splitext(os.path.basename(model_opts['model_gguf']))[0]
    else:
        print("Invalid model")
        continue

    if model is None:
        print("Error while loading model")
        continue

    dataset = load_dataset(DATASET_NAME)['train']
    data, embeddings, questions_embeddings = load_csv_data(os.path.join(DATA_EMBEDDINGS_PATH, model_opts['embeddings_id']), os.path.join(DATASET_EMBEDDINGS_PATH, model_opts['embeddings_id']))
    questions = find_questions_related_chunks(dataset, questions_embeddings, data, embeddings)
    top_k_info = get_top_k_scores_info(questions, data, embeddings)

    if 'model_id' in model_opts:
        model = ModelBuilder.get_from_id(model_opts['model_id'])
    elif 'model_gguf' in model_opts:
        model = ModelBuilder.get_from_gguf_file(model_opts['model_gguf'])
    else:
        print("Invalid model")
        continue

    responses_dir = os.path.join(RESPONSES_DIR, f'{model_opts['embeddings_id']}_{model_name}')
    os.makedirs(responses_dir, exist_ok=True)

    print(subprocess.run(['nvidia-smi']))

    if os.path.exists(os.path.join(responses_dir, 'predicted.csv')):
        reference = pd.read_csv(os.path.join(responses_dir, 'reference.csv'), sep=',', index_col=0)
        predicted = pd.read_csv(os.path.join(responses_dir, 'predicted.csv'), sep=',', index_col=0)
        contexts = pd.read_csv(os.path.join(responses_dir, 'context.csv'), sep=',', index_col=0)
    else:
        reference = pd.DataFrame(columns=['file', 'text'])
        predicted = pd.DataFrame(columns=['file', 'text'])
        contexts = pd.DataFrame(columns=['file', 'context'])
    q_total = len(questions)
    for q_idx, question in enumerate(questions):
        if q_idx < START:
            continue
        if END is not None and q_idx >= END:
            break

        print(f"{q_idx} - Question: {question['question']['question']}")

        if not question['question']['answers']['text'][0]:
            continue

        documents = []
        for doc_idx in top_k_info['embeddings_idx'][question['question']['id']]:
            data_doc = data.loc[doc_idx]
            doc = Document(
                content=data_doc['sentences'],
                metadata={
                    'document_name': data_doc['document_name'],
                    'title': data_doc['title'],
                    'path': data_doc['path'],
                    'parent': data_doc['parent'],
                },
            )
            documents.append(doc)

        try:
            response = model.query_with_documents(question['question']['question'], documents, add_to_history=False)
        except torch.OutOfMemoryError as e:
            print(e)
            for doc in documents:
                print("Document:", doc.print_to_console())
            continue

        if not response:
            continue

        print("Response:", response)

        # Appending model response
        res = {
            'id': question['question']['id'],
            'file': question['question']['title'],
            'text': response
        }
        df_res = pd.DataFrame(res, index=[q_idx])
        predicted = pd.concat([predicted, df_res])
        predicted.to_csv(os.path.join(responses_dir, 'predicted.csv'), sep=',')

        # Appending answer
        answ = {
            'id': question['question']['id'],
            'file': question['question']['title'],
            'text': question['question']['answers']['text'][0],
        }
        df_answ = pd.DataFrame(answ, index=[q_idx])
        reference = pd.concat([reference, df_answ])
        reference.to_csv(os.path.join(responses_dir, 'reference.csv'), sep=',')

        # Appending used context
        cntx = {
            'id': question['question']['id'],
            'file': question['question']['title'],
            'context': '|'.join([doc.metadata['document_name'] + ';' + doc.metadata['title'] for doc in documents])
        }
        df_cntx = pd.DataFrame(cntx, index=[q_idx])
        contexts = pd.concat([contexts, df_cntx])
        contexts.to_csv(os.path.join(responses_dir, 'context.csv'), sep=',')

        if q_idx % 100 == 0:
            print(subprocess.run(['nvidia-smi']))

    rouge_score = calculate_rouge(
        [res['text'] for _, res in predicted.iterrows()],
        [answ['text'] for _, answ in reference.iterrows()]
    )

    rouge = pd.DataFrame(rouge_score, index=[f"{model_opts['embeddings_id']}_{model_name}"])
    rouge_filename = os.path.join(RESULTS_DIR, f'rouge_top_{k}.csv')
    update_csv_data(rouge_filename, rouge)

    rouge_by_file= calculate_rouge_by_file(predicted, reference, model_opts, model_name)
    rouge_by_file_filename = os.path.join(RESULTS_DIR, f'rouge_by_file_top_{k}.csv')
    update_csv_data(rouge_by_file_filename, rouge_by_file, axis=1)

    del model
    torch.cuda.empty_cache()
