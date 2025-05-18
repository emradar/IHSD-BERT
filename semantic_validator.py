from sentence_transformers import SentenceTransformer, util
import spacy
import subprocess
import sys
from nltk.tokenize.treebank import TreebankWordDetokenizer


detokenizer = TreebankWordDetokenizer()

# Load sentence embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Ensure SpaCy model is available
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')

def is_augmentation_semantically_similar(original_sentence, original_word, candidate_word, threshold=0.75):
    original_sentence = detokenizer.detokenize(original_sentence)
    
    # Create augmented sentence with word replaced
    augmented_tokens = [
        candidate_word if token == original_word else token for token in original_sentence
    ]
    augmented_sentence = detokenizer.detokenize(augmented_tokens)

    # Get sentence embeddings
    emb1 = embedder.encode(original_sentence, convert_to_tensor=True)
    emb2 = embedder.encode(augmented_sentence, convert_to_tensor=True)
    
    similarity = util.pytorch_cos_sim(emb1, emb2)[0][0].item()
    return similarity >= float(threshold)

def pos_tag_text(text):
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]
