import pandas as pd
from datasets import Dataset, load_dataset
from sklearn.metrics import classification_report
from augment_dataset import augment_eval_dataset
from datasets import load_from_disk
import torch
from transformers import TrainingArguments, Trainer

def evaluate(name, model, tokenizer, percentage, compute_metrics):
    
    try:
        eval_dataset = load_from_disk(str(percentage)+"_augmented_implicit_hate_speech_dataset")

    except:
        try:
            eval_dataset = load_dataset("emradar/0.1_augmented_implicit_hate_speech_dataset")
        except:
            df = pd.read_csv("./implicit-hate-corpus/implicit_hate_v1_stg1_posts.tsv", delimiter='\t')

            def map_label(label):
                return 1 if label in ["explicit_hate", "implicit_hate"] else 0

            df["label"] = df["class"].map(map_label)
            eval_dataset = Dataset.from_pandas(df[["post", "label"]].rename(columns={"post": "text"}))
            eval_dataset = augment_eval_dataset(eval_dataset, percentage)


    eval_dataset = eval_dataset.map(
        lambda entry: tokenizer(entry["text"], padding="max_length", truncation=True, max_length=128),
        batched=True
    )

    eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    args = TrainingArguments(
        output_dir="./" + str(percentage) + "_" + name,
        per_device_eval_batch_size=32,
        dataloader_drop_last=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    predictions = trainer.predict(eval_dataset)
    preds = torch.tensor(predictions.predictions)
    preds = torch.argmax(preds, dim=-1)
    report = classification_report(eval_dataset["label"], preds, digits=4)

    with open(str(percentage) + "_" + name + ".txt", "w") as f:
        f.write(report)