import re
import phonetics
import nltk
from nltk.tokenize import word_tokenize
from transformers import pipeline, AlbertTokenizer, AlbertForSequenceClassification
import torch

# Download NLTK tokenizer
nltk.download("punkt")


def clean_word(word):
    """Remove non-alphabetic characters."""
    return re.sub(r'[^a-zA-Z]', '', word)


def phonetic_encode(word):
    """Convert a word to its phonetic representation."""
    cleaned_word = clean_word(word)
    if cleaned_word:
        return phonetics.soundex(cleaned_word)
    return None


# Load ALBERT for hate speech classification
MODEL_NAME = "albert_finetuned"  
tokenizer = AlbertTokenizer.from_pretrained(MODEL_NAME)
model = AlbertForSequenceClassification.from_pretrained(MODEL_NAME)


def classify_text_albert(text):
    """Classify text using ALBERT model."""                                                                                                                                          
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()

    return "hate speech" if prediction == 1 else "not hate speech"


def detect_hate_speech(text):
    words = word_tokenize(text.lower())  # Tokenize text
    phonetic_variants = [phonetic_encode(word) for word in words if phonetic_encode(word)]

    # Step 1: Phonetic matching with known hate speech patterns
    flagged_words = []
    known_hate_speech = {"H300", "NTS", "J200"}  # Add phonetic codes of known hate words

    for word, phonetic_code in zip(words, phonetic_variants):
        if phonetic_code in known_hate_speech:
            flagged_words.append(word)

    # Step 2: ALBERT classification
    albert_result = classify_text_albert(text)

    # Step 3: Decision logic
    if flagged_words or albert_result == "hate speech":
        return {"text": text, "flagged_words": flagged_words, "is_hate_speech": True}

    return {"text": text, "is_hate_speech": False}


# Example usage
example_texts = [
    "I h8 those ppl",
    "Wyte pipo be like...",
    "That dude is a total gr@pe",
]

for text in example_texts:
    result = detect_hate_speech(text)
    print(result)
