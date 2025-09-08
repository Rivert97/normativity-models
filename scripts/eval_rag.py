import os
import glob
import json

import dotenv
dotenv.load_dotenv()
import pandas as pd
import numpy as np
from sentence_transformers import util
import evaluate
import torch

from .lib.data import Document
from .lib.models import Builders

# Gobal variables
MODELS = [
    {'full_id': 'Qwen/Qwen3-Embedding-0.6B', 'id': 'QWEN', 'variant': '3-0.6B'},
    {'full_id': 'Qwen/Qwen3-Embedding-4B', 'id': 'QWEN', 'variant': '3-4B'},
    {'full_id': 'Qwen/Qwen3-Embedding-8B', 'id': 'QWEN', 'variant': '3-8B'},
]
DATA_PATH = './data'
DATASET_PATH = './dataset/ug-normativity-v0.1.json'
DATA_EMBEDDINGS_PATH = f'./data_embeddings'
DATASET_EMBEDDINGS_PATH = f'./dataset_embeddings'
DATASET_NAME = 'Rivert97/ug-normativity'
RESULTS_DIR = './results_rag'

# Other variables
k = 5
file_excludes = ['reglamento-de-responsabilidades-y-sanciones-en-materia-de-violencia-de-genero']
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

        data = pd.read_csv(data_filename, sep=',')
        embeddings = pd.read_csv(embeddings_filename, sep=',', index_col=0)
        questions_embeddings = pd.read_csv(questions_filename, sep=',', index_col=0)

        all_data.append(data)
        all_embeddings.append(embeddings)
        all_questions_embeddings.append(questions_embeddings)

    data = pd.concat(all_data, ignore_index=True)
    embeddings = pd.concat(all_embeddings, ignore_index=True)
    questions_embeddings = pd.concat(all_questions_embeddings)

    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    return data, embeddings, dataset, questions_embeddings

def find_questions_related_chunks(dataset, questions_embeddings, data, embeddings):
    # For each question find its chunk
    questions = []

    for question in dataset:
        if question['title'] in file_excludes:
            continue

        tmp_question = {
            'question': question,
            'question_embeddings': questions_embeddings.loc[question['id']].values,
            'chunk_idx': None,
            'chunk': None,
            'chunk_embeddings': None,
        }

        for chunk_idx, chunk in data.loc[data['document_name'] == question['title']].iterrows():
            if chunk['path'].lower().endswith(question['context'].lower().strip()):
                tmp_question['chunk_idx'] = chunk_idx
                tmp_question['chunk'] = chunk
                tmp_question['chunk_embeddings'] = embeddings.loc[chunk_idx].values
                break

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

def calculate_rouge(responses, answers):
    rouge = evaluate.load('rouge')
    score = rouge.compute(predictions=responses,
                            references=answers)

    return score

def calculate_rouge_by_file(responses, answers):
    files = list(set([res['file'] for res in responses]))
    metrics = {
        'rouge1': np.zeros((len(files),), dtype=np.float32),
        'rouge2': np.zeros((len(files),), dtype=np.float32),
        'rougeL': np.zeros((len(files),), dtype=np.float32),
        'rougeLsum': np.zeros((len(files),), dtype=np.float32),
    }
    for f_idx, f in enumerate(files):
        file_responses = filter(lambda r: r['file'] == f, responses)
        file_answers = filter(lambda a: a['file'] == f, answers)

        rouge = evaluate.load('rouge')
        score = rouge.compute(predictions=[r['text'] for r in file_responses],
                                references=[a['text'] for a in file_answers])

        metrics['rouge1'][f_idx] = score['rouge1']
        metrics['rouge2'][f_idx] = score['rouge2']
        metrics['rougeL'][f_idx] = score['rougeL']
        metrics['rougeLsum'][f_idx] = score['rougeLsum']

    return pd.DataFrame(metrics, index=files)

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
    print(f"Procesando {model_opts['full_id']}")
    data, embeddings, dataset, questions_embeddings = load_csv_data(os.path.join(DATA_EMBEDDINGS_PATH, model_opts['full_id']), os.path.join(DATASET_EMBEDDINGS_PATH, model_opts['full_id']))
    questions = find_questions_related_chunks(dataset, questions_embeddings, data, embeddings)
    top_k_info = get_top_k_scores_info(questions, data, embeddings)
    model = Builders[model_opts['id']].value.build_from_variant(model_opts['variant'])

    answers = []
    responses = []
    q_total = len(questions)
    for q_idx, question in enumerate(questions):
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
            print("Question:", question['question']['question'])
            for doc in documents:
                print("    import pdb; pdb.set_trace()Document:", doc.print_to_console())
            continue

        # Appending model response
        res = {
            'file': question['question']['title'],
            'text': response
        }
        responses.append(res)

        # Appending answer
        answ = {
            'file': question['question']['title'],
            'text': question['question']['answers']['text'][0],
        }
        answers.append(answ)

        print(f"Processed {q_idx+1}/{q_total}", end='\r')

    rouge_score = calculate_rouge(
        [res['text'] for res in responses],
        [answ['text'] for answ in answers]
    )

    rouge = pd.DataFrame(rouge_score, index=[model_opts['full_id']])
    rouge_filename = os.path.join(RESULTS_DIR, f'rouge_top_{k}.csv')
    update_csv_data(rouge_filename, rouge)

    rouge_by_file= calculate_rouge_by_file(responses, answers)
    rouge_by_file_filename = os.path.join(RESULTS_DIR, f'rouge_by_file_top_{k}.csv')
    update_csv_data(rouge_by_file_filename, rouge_by_file)

    del model
    torch.cuda.empty_cache()
