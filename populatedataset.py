from datasets import DatasetDict
import random
import math
from augmentation_functions import get_phonetically_similar_word, get_textually_similar_word, get_leetspeak_version, get_censored_word
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
#from semantic_validator import is_augmentation_semantically_similar

with open("bad_words.txt", "r", encoding="utf-8") as f:
    bad_words = [line.strip() for line in f if line.strip()]

def prepare_sentence(sentence):

    filtered_sentence = []

    # bad words are filtered since only these need to be augmented
    for word in sentence:
        if word.lower() in bad_words:
            filtered_sentence.append(word)

    return filtered_sentence


def augment_entry(entry, idx):

    # the entry is only augmented if it is explicitly hate speech
    # to not include cases that include bad words but are not hate speech
    if int(entry["hate_speech_score"] >= 0.5) == True:

        detokenizer = TreebankWordDetokenizer()

        original_sentence = word_tokenize(entry["text"])
        filtered_sentence = prepare_sentence(original_sentence) # only bad words
        augmented_sentence = []

        for word in original_sentence:
            if word in filtered_sentence:
                augmented_word = random.choice([get_phonetically_similar_word, get_textually_similar_word])(word)
                if augmented_word != word:
                    augmented_sentence.append(augmented_word)
                else:
                    augmented_word = random.choice([get_censored_word, get_leetspeak_version])(word)
                    augmented_sentence.append(augmented_word)
            else:
                augmented_sentence.append(word)

        return {"text": detokenizer.detokenize(augmented_sentence)}
    else: 
        return entry


def augment_dataset(dataset):
    
    # get indices for 10% of randomly selected entries
    percentage = 0.1
    train_dataset = dataset["train"]
    total_entries = len(train_dataset)
    total_entries_to_augment = math.floor(total_entries * percentage)
    selected_indices = random.sample(range(total_entries), total_entries_to_augment)

    # augment the selected entries
    changed_entries = train_dataset.select(selected_indices)
    augmented_entries = changed_entries.map(augment_entry, with_indices=True)

    # map changes to the original dataset with indices of changed_entries
    train_dictionary = train_dataset.to_dict()
    for idx, dataset_idx in enumerate(selected_indices):
        train_dictionary["text"][dataset_idx] = augmented_entries[idx]["text"]

    updated_train_dataset = train_dataset.from_dict(train_dictionary)
    updated_dataset = DatasetDict({"train": updated_train_dataset})

    updated_dataset.save_to_disk("augmented_hate_speech_dataset")

    return updated_dataset