import random
import difflib
import Levenshtein
import nltk
from nltk.corpus import cmudict, words


# a collection of the english vocabulary
try: 
    english_vocab = set(words.words()) 
except:
    nltk.download('words') 

# a collection of how they are pronounced
try:
    cmu = cmudict.dict()
except:
    nltk.download('cmudict') # words


def phoneme_distance(p1, p2):
    s1 = " ".join(p1)
    s2 = " ".join(p2)
    return Levenshtein.distance(s1, s2)


def get_phonetically_similar_word(word):
    """
    Get a phonetically close word

    The function first looks if the word exists in the CMU dictionary.
    Then it looks at other words in the dictionary to find the closest match
    through calculating the levenshtein distance.

    Returns the closest match as a str
    """

    word = word.lower()
    if word not in cmu:
        return word 

    word_pronounciation = cmu[word][0]
    candidates = []

    for candidate_word, pronunciations in cmu.items():
        if candidate_word == word:
            continue  # skip the original word
        for pronounciation in pronunciations:
            dist = phoneme_distance(word_pronounciation, pronounciation)
            candidates.append((candidate_word, dist))

    # sort candidates by levenshtein distance (lower is better)
    candidates.sort(key=lambda x: x[1])
    
    # return the closest candidate word only
    if candidates:
        return candidates[0][0]
    else:
        return word  
    

def get_textually_similar_word(word):
    """
    Get a textually close word

    The function looks at the words in the english vocabulary through "words" in "nltk.corpus" 
    and returns the closest match with "difflib".
    """
    candidates = difflib.get_close_matches(word.lower(), english_vocab, 1, 0.75)

    return candidates[0] if candidates else word  


vowels = {'a', 'e', 'i', 'o', 'u'}


def get_leetspeak_version(word):
    """
    Get the leetspeak version of a word

    The function changes random characters that are often used in leetspeak
    Vowels have a higher probablity to be replaced (0.8)
    """
    obfuscation = {
        'a': ['@', '4'],
        'b': ['8', '6'],
        'e': ['3'],
        'i': ['1', '!'],
        'l': ['1', '|'],
        'o': ['0'],
        's': ['$', '5'],
        't': ['7', '+'],
    }

    augmented_word = ""

    for char in word.lower():

        if char in obfuscation:
            change_probability = 0.8 if char in vowels else 0.5
            if random.random() < change_probability:
                augmented_word += random.choice(obfuscation[char])
            else:
                augmented_word += char
        else:
            augmented_word += char

    return augmented_word


def get_censored_word(word):
    """
    Get the censored version of a word

    The function censors vowels with an asterisk (*) to simulate cases when users selfcensor
    """

    augmented_word = ""

    for char in word.lower():
        if char in vowels:
            augmented_word += '*'
        else:
            augmented_word += char

    return augmented_word


def printresult(word):
    """
    Prints the result from the functions

    This function is only meant for debugging
    """
    print(get_censored_word(word))
    print(get_leetspeak_version(word))
    print(get_phonetically_similar_word(word))
    print(get_textually_similar_word(word))