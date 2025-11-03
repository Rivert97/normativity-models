import logging
import sys

from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

DATASET_NAME = "Rivert97/ug-normativity"
BATCH_SIZE = 16

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

# You can specify any Hugging Face pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = sys.argv[1] if len(sys.argv) > 1 else "all-MiniLM-L6-v2"
model_name_only = model_name.split("/")[-1]

# 1. Load a model to finetune with 2. (Optional) model card data
model = SentenceTransformer(
    model_name,
)

# Create a LoRA adapter for the model
peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    inference_mode=False,
    r=64,
    lora_alpha=128,
    lora_dropout=0.1,
)
model.add_adapter(peft_config)

# 3. Load a dataset to finetune on
dataset = load_dataset(DATASET_NAME, split="train")
dataset_dict = dataset.train_test_split(test_size=0.2, seed=12)
train_dataset: Dataset = dataset_dict["train"]
eval_dataset: Dataset = dataset_dict["test"]

# 4. Define a loss function
loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=32)

# 5. (Optional) Specify training arguments
run_name = f"{model_name_only}-ug-normativity-peft"
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=f"models/{run_name}",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=True,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=25,
    logging_first_step=True,
    run_name=run_name,  # Will be used in W&B if `wandb` is installed
)

# 7. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
)
trainer.train()

# 8. Save the trained model
final_output_dir = f"models/{run_name}/final"
model.save_pretrained(final_output_dir)
