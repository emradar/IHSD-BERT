import torch
from datasets import load_dataset
from transformers import AlbertTokenizer,AlbertForSequenceClassification,TrainingArguments,Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from itertools import product

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech")

# Convert to binary labels
def to_binary(example):
    example["label"] = int(example["hate_speech_score"] >= 0.5)
    return example

dataset = dataset["train"].map(to_binary)

# Split into train/validation
split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
val_dataset = split["test"]

# Tokenizer
tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

columns = ["input_ids", "attention_mask", "label"]
train_dataset.set_format(type="torch", columns=columns)
val_dataset.set_format(type="torch", columns=columns)

# Compute metrics
def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    labels = pred.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Hyperparameter search
lr = 2e-5
bs = 32
wd = 0.01
epoch = 2
results = []

name = f"e{epoch}lr{lr}_bs{bs}_wd{wd}".replace(".", "")
print(f"\nTraining {name}...")

training_args = TrainingArguments(
    output_dir=f"./results/{name}",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=epoch,
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs,
    learning_rate=lr,
    weight_decay=wd,
    fp16=torch.cuda.is_available(),
    logging_dir=f"./logs/{name}",
    logging_strategy="steps",
    logging_steps=50,
    report_to="tensorboard",
    disable_tqdm=False,
)

model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=2).to(device)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
metrics = trainer.evaluate()

print(metrics)