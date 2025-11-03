import dotenv
dotenv.load_dotenv()

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from datasets import load_dataset

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DATASET_NAME = 'Rivert97/ug-normativity'
SAVE_MODEL = './Rivert97/all-MiniLM-L6-v2-ug-normativity'
BATCH_SIZE = 16
EPOCHS = 2
WARMUP_STEPS = 10

# Load base model
model = SentenceTransformer(MODEL_ID)

# load dataset
dataset = load_dataset(DATASET_NAME, split='train')
print(dataset)

train = []
for data in dataset:
    train.append(InputExample(texts=[data['question'], data['context_text']]))

train_dataloader = DataLoader(train, shuffle=True, batch_size=BATCH_SIZE)

# Use MultipleNegativesRankingLoss for contrastive fine-tuning
train_loss = losses.MultipleNegativesRankingLoss(model)

# Train
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=WARMUP_STEPS,
    show_progress_bar=True
)

model.save(SAVE_MODEL)