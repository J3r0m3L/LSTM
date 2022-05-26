# https://www.youtube.com/watch?v=VAMKuRAh2nc&t=230s&ab_channel=KGPTalkie
# Need to scale this model down because I keep on getting 'Dst tensor is not initialized'
# https://stackoverflow.com/questions/37313818/tensorflow-dst-tensor-is-not-initialized
import tensorflow as tf
import string
import requests as re
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences

response = re.get('https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt')
data = response.text.split('\n')[253:]
data = " ".join(data)

def clean_text(doc):
    tokens = doc.split(" ") # splits every word via spacing
    table = str.maketrans('', '', string.punctuation) # remove punctuation
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()] # remove numbers
    tokens = [word.lower() for word in tokens] # make everything lowercase
    return tokens

tokens = clean_text(data)
#print(len(tokens)) #898199 words
#print(len(set(tokens))) #27956 words

length = 50 + 1 # check first 50 and predict Last 1 
lines = []

for i in range(length, len(tokens)):
    seq = tokens[i-length:i]
    lines.append(' '.join(seq))

    # too many words so break before crash (200000) is too much 20000 works though
    if i > 20000:
        break

# prepare data and labels
tokenizer = Tokenizer() # vectorizes text into vector where the coefficient for each token could be binary
tokenizer.fit_on_texts(lines) # updates internal vocabulary based off of lines (returns object)
sequences = np.array(tokenizer.texts_to_sequences(lines)) # transforms texts into a sequence of integers and places in np array (199951, 51)

x, y = sequences[:, :-1], sequences[:, -1] #train 'features': (199951, 50) and train 'labels': (199951, 1)
vocab_size = len(tokenizer.word_index) + 1 #13009
y = to_categorical(y, num_classes=vocab_size) #converts a class vector (integers) to binary class matrix (one hot encoding)
seq_length = x.shape[1] # literally just 50

# Actual LSTM Model
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length)) # turns positive integers (indexes) into dense vectors of fixed size
model.add(LSTM(100, return_sequences=True)) # return the last output
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, batch_size = 56, epochs=100) # 'DST tensor not initialized.' Stackoverflow says batchsize is to large for memory (original: 256)
model.save('lstm_nlp_model') # save model becauset this takes forever to train

