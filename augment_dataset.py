from datasets import DatasetDict, Dataset
import random
import math
import requests
import time
import datetime
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from augmentation_functions import (
    get_phonetically_similar_word, 
    get_textually_similar_word, 
    get_leetspeak_version, 
    get_censored_word
)

total_augments = 0

# opening the list of known bad words
try:
    with open("bad_words.txt", "r", encoding="utf-8") as f:
        bad_words = [line.strip() for line in f if line.strip()]
except:
    # if the file can't be opened, a fall-back list is used instead
    bad_words = requests.get("https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/en").text.splitlines()

def prepare_sentence(sentence):
    """
    Prepare the sentence before working on it

    The function checks for bad words in the sentence and returns a list of tokens containing the bad words.

    Bad words are filtered since only these need to be augmented
    """
    filtered_sentence = []

    for word in sentence:
        if word.lower() in bad_words:
            filtered_sentence.append(word)

    return filtered_sentence


def augment_entry(entry, idx):
    """
    Augment an entry in the dataset

    The function augments the entry randomly based on different conditions.
    The entry is only augmented if it is explicit hate speech and includes bad words.
    This is to not include cases that include bad words but are not hate speech.
    """
    global total_augments
    detokenizer = TreebankWordDetokenizer()
    original_sentence = word_tokenize(entry["text"])
    filtered_sentence = prepare_sentence(original_sentence) # only bad words

    if len(filtered_sentence) > 0:

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

        total_augments += 1
        augmented_entry = dict(entry)
        augmented_entry["text"] = detokenizer.detokenize(augmented_sentence)
        return augmented_entry
    else: 

        return entry


def augment_dataset(dataset: DatasetDict, percentage: float) -> DatasetDict:
    """
    Augment the given dataset

    The augmentations are done on a given percentage of the entries labeled as hate speech
    (see augment_entry for more details)

    The entries and the changes are logged in "augmentation_log.txt"
    The dataset is saved in "augmented_hate_speech_dataset"

    Returns the updated dataset
    """
    start_time = time.time()
    timestamp = datetime.datetime.now()

    # get 10% of randomly selected hate speech entries
    train_dataset = dataset["train"]
    hate_speech_entries = train_dataset.filter(lambda entry: entry["hate_speech_score"] >= 0)
    
    # sample a percentage of the hate speech sample
    num_hate_speech_entries = len(hate_speech_entries)
    total_entries_to_augment = math.floor(num_hate_speech_entries * percentage)
    selected_indices = random.sample(range(num_hate_speech_entries), total_entries_to_augment)

    # augment the selected entries
    changed_entries = train_dataset.select(selected_indices)
    augmented_entries = changed_entries.map(augment_entry, with_indices=True)

    # map changes to the original dataset with indices of changed_entries
    train_dictionary = train_dataset.to_dict()

    log_lines = []

    for idx, dataset_idx in enumerate(selected_indices):
        original = changed_entries[idx]["text"]
        augmented = augmented_entries[idx]["text"]
        train_dictionary["text"][dataset_idx] = augmented

        # log of changes
        log_lines.append(
            f"Index {dataset_idx}:\nOriginal: {original}\nAugmented: {augmented}\n---\n"
        )

    updated_train_dataset = train_dataset.from_dict(train_dictionary)
    updated_dataset = DatasetDict({"train": updated_train_dataset})
    updated_dataset.save_to_disk(str(percentage)+"_augmented_hate_speech_dataset")

    end_time = time.time()
    duration = round(end_time - start_time, 2)

    log_header = (
        f"Augmentation Run Timestamp: {timestamp}\n"
        f"Total Entries: {num_hate_speech_entries}\n"
        f"Total entries augmented: {total_augments}"
        f"Augmented Entries: {total_entries_to_augment}\n"
        f"Time Taken: {duration / 60} minutes\n"
        f"{'-'*40}\n\n"
    )

    with open("augmentation_log.txt", "w", encoding="utf-8") as f:
        f.write(log_header)
        f.writelines(log_lines)

    return updated_dataset

def augment_eval_dataset(dataset: Dataset, percentage: float) -> Dataset:
    """
    Augment the given dataset

    The augmentations are done on a given percentage of the entries labeled as hate speech
    I.e. when "label" == 1

    The entries and the changes are logged in "augmentation_log.txt"
    The dataset is saved in f"{percentage}_augmented_implicit_hate_speech_dataset"

    Returns the augmented dataset.
    """
    start_time = time.time()
    timestamp = datetime.datetime.now()

    # filter for hate speech entries
    hate_speech_entries = dataset.filter(lambda entry: entry["label"] == 1)

    num_hate_speech_entries = len(hate_speech_entries)
    total_entries_to_augment = math.floor(num_hate_speech_entries * percentage)
    selected_indices = random.sample(range(num_hate_speech_entries), total_entries_to_augment)

    # select and augment entries
    changed_entries = hate_speech_entries.select(selected_indices)
    augmented_entries = changed_entries.map(augment_entry, with_indices=True)

    # convert original dataset to a dictionary for in-place update
    dataset_dict = dataset.to_dict()

    log_lines = []

    for idx, dataset_idx in enumerate(selected_indices):
        original = changed_entries[idx]["text"]
        augmented = augmented_entries[idx]["text"]
        dataset_dict["text"][dataset_idx] = augmented

        log_lines.append(
            f"Index {dataset_idx}:\nOriginal: {original}\nAugmented: {augmented}\n---\n"
        )

    # create updated dataset
    updated_dataset = Dataset.from_dict(dataset_dict)
    updated_dataset.save_to_disk(f"{percentage}_augmented_implicit_hate_speech_dataset")

    end_time = time.time()
    duration = round(end_time - start_time, 2)

    log_header = (
        f"Augmentation Run Timestamp: {timestamp}\n"
        f"Total Entries: {num_hate_speech_entries}\n"
        f"Total entries augmented: {total_entries_to_augment}\n"
        f"Time Taken: {duration / 60} minutes\n"
        f"{'-'*40}\n\n"
    )

    with open("augmentation_log.txt", "w", encoding="utf-8") as f:
        f.write(log_header)
        f.writelines(log_lines)

    return updated_dataset