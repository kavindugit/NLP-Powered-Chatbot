import json
import pickle
import random
import nltk
import numpy as np
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

with open("chatbot_model.pkl" , "rb") as f :
    model = pickle.load(f)

with open("words.pkl" , "rb") as f:
    words , labels = pickle.load(f)

with open("intents.json") as f:
    intents = json.load(f)

#process user input
def preprocess(sentence):
    tokens = nltk.word_tokenize(sentence)
    tokens = [stemmer.stem(w.lower()) for w in tokens if w.isalpha()]
    bag = [0] * len(words)
    for token in tokens:
        for i , word in enumerate(words):
            if word == token:
                bag[i] = 1
    return np.array(bag).reshape(1,-1)

#Predict and get response:
def get_response(user_input):
    input_vector = preprocess(user_input)
    predicted_index = model.predict(input_vector)[0]
    predicted_tag = labels[predicted_index]

    for intent in intents["intents"]:
        if intent["tag"] == predicted_tag:
            return random.choice(intent["responses"])
    return "Sorry I didnt understand that"

print("Pybot : hellow! (type 'quit' to exit)")

while True:
    msg = input("You : ")
    if msg.lower() =="quit":
        print("PyBot : Goodbye!")
        break
    response = get_response(msg)
    print("pybot : ", response)
