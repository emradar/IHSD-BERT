#pip install transformers datasets torch scikit-learn pandas

import pandas as pd
from datasets import Dataset

# Load dataset from CSV
df = pd.read_csv("final_hate_speech_dataset.csv")

# Ensure correct column names
df = df.rename(columns={"tweet": "text", "class": "label"})

# Convert to Hugging Face Dataset format
dataset = Dataset.from_pandas(df)

dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

from transformers import AlbertTokenizer, AlbertForSequenceClassification

MODEL_NAME = "albert-base-v2"

# Load tokenizer & model
tokenizer = AlbertTokenizer.from_pretrained(MODEL_NAME)
model = AlbertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)  # 2 labels (hate / non-hate)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Define evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Training settings
training_args = TrainingArguments(
    output_dir="./albert_hate_speech",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,  # Increase if needed
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# Save model
model.save_pretrained("fine_tuned_albert_hate")
tokenizer.save_pretrained("fine_tuned_albert_hate")

# Load saved model
from transformers import AlbertForSequenceClassification

fine_tuned_model = AlbertForSequenceClassification.from_pretrained("fine_tuned_albert_hate")

def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = fine_tuned_model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return "Hate Speech" if prediction == 1 else "Not Hate Speech"

# Test example
print(classify_text("I h8 those ppl"))
print(classify_text("Have a nice day!"))

from transformers import Trainer, TrainingArguments
import itertools

# Define hyperparameter search space
batch_sizes = [8, 16, 32]
learning_rates = [2e-5, 3e-5, 5e-5]
epochs = [3, 5, 7]

# Store results
best_model = None
best_accuracy = 0
best_params = None

# Try different combinations
for batch_size, lr, epoch in itertools.product(batch_sizes, learning_rates, epochs):
    print(f"🔍 Testing batch={batch_size}, lr={lr}, epochs={epoch}")

    training_args = TrainingArguments(
        output_dir="./albert_hate_speech",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epoch,
        weight_decay=0.01,
        learning_rate=lr,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluate model
    results = trainer.evaluate()
    accuracy = results["eval_accuracy"]

    # Save best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = (batch_size, lr, epoch)
        model.save_pretrained("best_albert_hate_model")
        tokenizer.save_pretrained("best_albert_hate_model")

print(
    f"✅ Best Model: batch={best_params[0]}, lr={best_params[1]}, epochs={best_params[2]}, accuracy={best_accuracy:.4f}")
