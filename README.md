# Implicit Hate Speech Detection with ALBERT
This repository consists of code used to develop a NLP model to detect implicit forms of hate speech. 
It is part of research done by Emir Adar and Alma Wiberg (2025).

The trained model is uploaded on [Hugging Face](https://huggingface.co/emradar/IHSD-BERT).

## Prerequisites
The code is run in Virtual Studio Code with the Python 3.12.10 interpreter and packages included in [requirements.txt](requirements.txt).

## Data augmentation
The data used to train the model is based on [measuring-hate-speech](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) on work done by Kennedy et al., (2020) and Sachdeva
et al., (2022) and contains over 39,000 comments annotated by nearly 8,000 individuals, resulting in 135,000+ entries. 

Part of this data (up to 10%) is then augmented to include common tactics to avoid hate speech detection. 
This includes leetspeak, self-censoring, and phonetic and syntactic similarities between words. 
All of these are implemented in [augmentation_functions.py](augmentation_functions.py) and used in [augment_dataset.py](augment_dataset.py).
Only entries that are explicitly hate speech in the dataset and include bad words that are given by 
[bad_words.txt](bad_words.txt) are augmented. This is to not include cases where bad words are used but are not hate speech. 
The motivation for that is that people are more likely to try to evade hate speech detection when they are hateful.

The list of bad words are derived from [LDNOOBW](https://github.com/LDNOOBW) and edited to fit our purpose. 
This includes removing phrases, emojis and words that are uncommon in hate speech.

The augmented dataset is then saved in a separate file called [augmented_hate_speech_dataset](augmented_hate_speech_dataset) and the changes are logged in [augmentation_log.txt](augmentation_log.txt). It is also saved on [Hugging Face](https://huggingface.co/datasets/emradar/0.1_augmented_hate_speech_dataset).

## Fine-tuning
The code for fine-tuning is found in [train_albert.py](train_albert.py).

## Evaluation
The model was evaluated using an augmented version of [SALT-NLP/implicit-hate](https://github.com/SALT-NLP/implicit-hate) that is saved on [Hugging Face](https://huggingface.co/datasets/emradar/0.1_augmented_implicit_hate_speech_dataset).

The code for evaluating the models is seen in [evaluation.py](evaluation.py).

## Results
The results for each model is seen in [0.1_IHSD-BERT.txt](0.1_IHSD-BERT.txt) and [0.1_hateBERT.txt](0.1_hateBERT.txt).

