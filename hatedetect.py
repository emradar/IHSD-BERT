import pandas as pd
import torch
from transformers import AlbertTokenizer, AlbertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from fastapi import FastAPI
import uvicorn
import itertools


# 1. Load and preprocess dataset
def load_and_preprocess_data():
    df = pd.read_csv("final_hate_speech_dataset.csv")  # Replace with your dataset path
    df = df.rename(columns={"tweet": "text", "class": "label"})  # Ensure correct column names
    dataset = Dataset.from_pandas(df)

    # Split dataset into train and test sets
    dataset = dataset.train_test_split(test_size=0.2)
    return dataset["train"], dataset["test"]


# 2. Tokenize the data
def tokenize_data(train_dataset, eval_dataset):
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    return train_dataset, eval_dataset, tokenizer


# 3. Hyperparameter tuning & model training
def train_and_tune_hyperparameters(train_dataset, eval_dataset, tokenizer):
    best_model = None
    best_accuracy = 0
    best_params = None
    batch_sizes = [8, 16, 32]
    learning_rates = [2e-5, 3e-5, 5e-5]
    epochs = [3, 5, 7]

    # Evaluation metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = torch.argmax(logits, axis=1).cpu().numpy()
        labels = labels.cpu().numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

    # Hyperparameter search space
    for batch_size, lr, epoch in itertools.product(batch_sizes, learning_rates, epochs):
        print(f"Testing batch={batch_size}, lr={lr}, epochs={epoch}")

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

        model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=2)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        # Save model if it performs better
        results = trainer.evaluate()
        accuracy = results["eval_accuracy"]
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = (batch_size, lr, epoch)
            model.save_pretrained("best_albert_hate_model")
            tokenizer.save_pretrained("best_albert_hate_model")

    print(
        f"Best Model: batch={best_params[0]}, lr={best_params[1]}, epochs={best_params[2]}, accuracy={best_accuracy:.4f}")
    return model, tokenizer


# 4. FastAPI app for deployment
def create_fastapi_app(model, tokenizer):
    app = FastAPI()

    @app.post("/predict/")
    async def predict(text: str):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        return {"text": text, "prediction": "Hate Speech" if prediction == 1 else "Not Hate Speech"}

    return app


# 5. Run the application
def run_api():
    train_dataset, eval_dataset = load_and_preprocess_data()
    train_dataset, eval_dataset, tokenizer = tokenize_data(train_dataset, eval_dataset)
    model, tokenizer = train_and_tune_hyperparameters(train_dataset, eval_dataset, tokenizer)

    app = create_fastapi_app(model, tokenizer)

    # Run FastAPI server with Uvicorn (use --reload for hot-reloading)
    uvicorn.run(app, host="127.0.0.1", port=8000)


# Run the entire pipeline
if __name__ == "__main__":
    run_api()
