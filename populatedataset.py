from random import randint
from datasets import load_dataset
import phonetics
import random
import math
from nltk.tokenize import word_tokenize
from PyDictionary import PyDictionary

"""TODO add random amount of characters to be obfuscated"""
def get_leetspeech_version(word):
    obfuscation = {
        'a': ['@', '4'],
        'b': ['8', '6'],
        'e': ['3'],
        'i': ['1', '!'],
        'l': ['1', '|'],
        'o': ['0'],
        's': ['$', '5'],
        't': ['7', '+'],
        'ate': ['8']
    }

    result = ""

    if 'ate' in word:
        word.replace()
    for char in word.lower():
        if char in obfuscation:
            result += random.choice(obfuscation[char])
        else:
            result += char

    return result


def get_textually_similar_word(word):
    # use a word dictionary
    # compare the textual similarity to other words in the dictionary
    # randomly substitute with the closest match


def get_semantically_similar_word(word):
    # use PyDictionary
    # compare the meaning of the word to other words in the dictionary
    # randomly substitute with the closest match


def get_phonetically_similar_word(word):

    # use dictionary made through wordphonetics.py
    # compare phonetics with other words in the dictionary
    # randomly substitute with closest match
    soundex = phonetics.soundex(word)
    phonetics.soundex

def augment_entry(index):

    sentence = word_tokenize(dataset["train"]["text"][index])

    max_index = len(sentence)

    # for each word in the entry substitute a random amount of words with
        # get_leetspeech_version
        # get_textually_similar_word
        # get_semantically_similar_word
        # get_phonetically_similar_word

    selected_indices = random.sample(0, max_index) # random indices of words
    
    for word in sentence:
        random.choice(selected_indices)

def augment_dataset(dataset):

    # get indices for 10% of randomly selected entries
    total_entries_to_augment = math.floor(dataset.shape['train'][0]* 0.1)
    
    selected_indices = random.sample(range(0, total_entries_to_augment))

    # augment the selected entries
    changed_entries = dataset.select(selected_indices)

    augmented_entries = changed_entries.map(augment_entry)

    # map changes to the original dataset with indices of changed_entries
    
    
    dataset.map(augment_entry, with_indices=True)
    
    
    return {"full_dataset": dataset, "augmented_subset": augmented_entries}

dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech")