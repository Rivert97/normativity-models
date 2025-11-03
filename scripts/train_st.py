import dotenv
dotenv.load_dotenv()

import torch.distributed as dist
from torch.distributed import get_rank, is_initialized
from sentence_transformers import SentenceTransformer, losses, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from datasets import load_dataset, Dataset

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DATASET_NAME = 'Rivert97/ug-normativity'
SAVE_MODEL = './Rivert97/all-MiniLM-L6-v2-ug-normativity'
BATCH_SIZE = 16
EPOCHS = 2
WARMUP_STEPS = 10

def main():
    # 1. Load base model
    model = SentenceTransformer(MODEL_ID)

    # 2. load training dataset
    dataset = load_dataset(DATASET_NAME, split='train')
    print(dataset)

    train_anchors = [data['question'] for data in dataset]
    train_positives = [data['context_text'] for data in dataset]
    train_dataset = Dataset.from_dict({
        "anchor": train_anchors,
        "positive": train_positives,
    })

    # 3. Define a loss function
    # Use MultipleNegativesRankingLoss for contrastive fine-tuning
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # 4. Define training arguments
    args = SentenceTransformerTrainingArguments(
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        fp16=True,
        dataloader_drop_last=True,
    )

    # 5. Train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,
    )
    trainer.train()

    if not is_initialized() or get_rank() == 0:
        trainer.model.save(SAVE_MODEL)

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()