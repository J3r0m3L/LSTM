# https://www.youtube.com/watch?v=VAMKuRAh2nc&t=230s&ab_channel=KGPTalkie
# Need to scale this model down because I keep on getting 'Dst tensor is not initialized'
# https://stackoverflow.com/questions/37313818/tensorflow-dst-tensor-is-not-initialized
import tensorflow as tf
import string 
import requests

import numpy as np 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences

response = requests.get('https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt')
data = response.text.split('\n')
data = data[253:]
data = " ".join(data)

def clean_text(doc):
    tokens = doc.split(" ")
    table = str.maketrans('', '',string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    return tokens

tokens = clean_text(data)
# print(len(tokens)) Number of Words
# print(len(set(tokens))) Number of Unique Words

length = 50 + 1
lines = []

for i in range(length, len(tokens)):
    seq = tokens[i-length:i]
    line = ' '.join(seq)
    lines.append(line)

    # There are too many words so we are going to break before that happens
    if i > 200000:
        break

# Prepare Data and Labels
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)

sequences = np.array(sequences)
X, Y = sequences[:, :-1], sequences[:,-1]
vocab_size = len(tokenizer.word_index) + 1
Y = to_categorical(Y, num_classes=vocab_size)
seq_length = X.shape[1]
#print(len(tokenizer.word_index))
# print(X[0])
# print(Y[0])

# Actual LSTM Model
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
#print(model.summary())

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, batch_size = 56, epochs=100) # Gets an error if we run a batch size 256

# Predict
seed_text = lines[12343]
def generate_text_seq(model, tokenizer, text_seq_length, seed_text, n_words):
    text = []
    for _ in range(n_words):
        encoded = tokenizer.texts_to_sequences([seed_text])[0]
        encoded = pad_sequences([encoded], maxlen = text_seq_length, truncating = 'pre')

        y_predict = model.predict_classes(encoded)
        predicted_word = ''
        for word, index in tokenizer.word_index.items():
            if index == y_predict: 
                predicted_word = word
                break
                
        seed_text = seed_text + ' ' + predicted_word
        text.append(predicted_word)
    return ' '.join(text)

# if works should predict the next 10 words of the seed text
print(generate_text_seq(model, tokenizer, seq_length, seed_text, 10))
print('-------------------------------------')
print(seed_text)

