import torch
import tensorflow as tf
from datasets import load_dataset
from transformers import AlbertTokenizer, AlbertForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datetime import datetime

#checking for GPU availability for faster training speeds
print(torch.cuda.is_available()) 
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
print(tf.config.list_physical_devices('GPU'))

# loading dataset
dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech")

# converting hate speech score into binary labels
def to_binary(example):
    example["label"] = int(example["hate_speech_score"] >= 0.5)
    return example

# transformations on the dataset
def transform_dataset():
    dataset = dataset["train"].map(to_binary)
    

# splitting dataset into train and validation
split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
val_dataset = split["test"]

# tokenization
tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# setting format for PyTorch
columns = ["input_ids", "attention_mask", "label"]
train_dataset.set_format(type="torch", columns=columns)
val_dataset.set_format(type="torch", columns=columns)

# loading ALBERT model
model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

# metrics
def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    labels = pred.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# training arguments
log_dir = f"./logs/albert_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
training_args = TrainingArguments(
    output_dir="./albert_finetuned",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir=log_dir,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    logging_strategy="steps",
    logging_steps=50,
    report_to="tensorboard",  
)

# trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# training the model
trainer.train()