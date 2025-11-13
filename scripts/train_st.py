import sys
import logging
import random

import dotenv
dotenv.load_dotenv()

import torch.distributed as dist
from torch.distributed import get_rank, is_initialized
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from transformers import EarlyStoppingCallback
from datasets import load_dataset, Dataset

DEFAULT_MODEL = 'all-MiniLM-L6-v2'
DATASET_NAME = 'Rivert97/ug-normativity'
SAVE_MODEL = './Rivert97/'
BATCH_SIZE = 32
EPOCHS = 2
TRIPLETS = True

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

# You can specify any Hugging Face pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
model_name_only = model_name.split("/")[-1]

def main():
    # 1. Load base model
    model = SentenceTransformer(model_name)

    # 2. Load training dataset
    dataset = load_dataset(DATASET_NAME, split='train')
    dataset_dict = dataset.train_test_split(test_size=0.1, seed=12)
    if TRIPLETS:
        train_dataset: Dataset = Dataset.from_dict({"anchor": dataset_dict["train"]["question"], "positive": dataset_dict["train"]["context_text"], "negative": random.sample(dataset_dict["train"]["context_text"], len(dataset_dict["train"]["context_text"]))})
    else:
        train_dataset: Dataset = Dataset.from_dict({"anchor": dataset_dict["train"]["question"], "positive": dataset_dict["train"]["context_text"]})
    print(train_dataset)

    # 3. Define a loss function
    train_loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=32)

    # 4. Define training arguments
    if TRIPLETS:
        run_name = f"{model_name_only}-ug-normativity-triplets"
    else:
        run_name = f"{model_name_only}-ug-normativity"
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=f"models/{run_name}",
        # Optional training parameters:
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        logging_steps=25,
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # 5. Create an evaluator & evaluate the base model
    print("Initial evaluation")
    dev_evaluator = InformationRetrievalEvaluator(
        queries={q["id"]: q["question"] for q in dataset_dict["test"]},
        corpus={"Doc_"+q["id"]: q["context_text"] for q in dataset_dict["test"]},
        relevant_docs={q["id"]:["Doc_"+q["id"]] for q in dataset_dict["test"]},
        batch_size=BATCH_SIZE,
        name="sts-dev",
    )
    dev_evaluator(model)

    # 6. Train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=Dataset.from_dict({"anchor": dataset_dict["test"]["question"], "positive": dataset_dict["test"]["context_text"], "negative": random.sample(dataset_dict["test"]["context_text"], len(dataset_dict["test"]["context_text"]))}),
        loss=train_loss,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    trainer.train()

    print("Final evaluation")
    dev_evaluator(model)

    # 7. Evaluate with test dataset
    print("Test evaluation")
    test_dataset = load_dataset(DATASET_NAME, split='test')
    print(test_dataset)
    test_evaluator = InformationRetrievalEvaluator(
        queries={q["id"]: q["question"] for q in test_dataset},
        corpus={"Doc_"+q["id"]: q["context_text"] for q in test_dataset},
        relevant_docs={q["id"]:["Doc_"+q["id"]] for q in test_dataset},
        batch_size=BATCH_SIZE,
        name="sts-test",
    )
    test_evaluator(model)

    if not is_initialized() or get_rank() == 0:
        trainer.model.save(f"{SAVE_MODEL}/{run_name}")

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
