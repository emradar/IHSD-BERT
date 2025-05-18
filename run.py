import torch
from transformers import AlbertTokenizer, AlbertForSequenceClassification

# tokenization
tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

# loading ALBERT model
#model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=2)

model = AlbertForSequenceClassification.from_pretrained("./albert_finetuned/checkpoint-6778", num_labels=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

text = input("Write a sentence!")  # coded or obfuscated hate

inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    print("Prediction:", "Hate Speech" if prediction == 1 else "Not Hate Speech")
