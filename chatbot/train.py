import nltk
import json
import random
import pickle
import numpy as np
from nltk.stem import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
nltk.download('punkt')
nltk.download('punkt_tab')

# NLP tool

stemmer = PorterStemmer()

# load intents

with open("intents.json") as file:
    data = json.load(file)

words = []    # all stemmed words in patterns
labels =[]   # Tags
documents =[]  # (tokenized pattern , tag)

for intent in data ["intents"]:
    for pattern in intent["patterns"]:
        # tokenize each word in the sentence
        tokenized = nltk.word_tokenize(pattern)
        # stem each word and convert to lowercase
        words.extend(tokenized)
        # add to documents
        documents.append((tokenized, intent["tag"]))

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

# stem and lower each word and remove duplicates
words =[stemmer.stem(w.lower()) for w in words if w.isalpha()]
# remove duplicates
words = sorted(set(words))
# sort labels
labels = sorted(labels)

X_train = []
y_train = []

for(pattern_words , tag) in documents:
    bag = [0] * len(words)
    pattern_words = [stemmer.stem(w.lower()) for w in pattern_words]

    for word in pattern_words:
        for i , w in enumerate(words):
            if w == word:
                bag[i] = 1

    X_train.append(bag)
    y_train.append(labels.index(tag))

model = MultinomialNB()
model.fit(X_train , y_train)

with open("chatbot_model.pkl" , "wb") as f:
    pickle.dump(model , f)
with open("words.pkl", "wb") as f:
    pickle.dump((words , labels) , f)

print("Model trained and saved")






