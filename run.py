from train_albert import train_model, compute_metrics
from evaluation import evaluate
from transformers import AlbertTokenizer, AlbertForSequenceClassification, AutoTokenizer, BertForSequenceClassification

# train and evaluate our own model
#train_model("IHSD-BERT", 0.1)

# evaluate our model and hateBERT
tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
try:
    model = AlbertForSequenceClassification.from_pretrained("./0.1_IHSD-BERT/checkpoint-10167", num_labels=2)
except:
    model = AlbertForSequenceClassification.from_pretrained("emradar/IHSD-BERT/0.1_IHSD-BERT/checkpoint-10167", num_labels=2)
evaluate("IHSD-BERT", model, tokenizer, 0.1, compute_metrics)

tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT")
model = BertForSequenceClassification.from_pretrained("GroNLP/hateBERT", num_labels=2)
evaluate("hateBERT", model, tokenizer, 0.1, compute_metrics)
