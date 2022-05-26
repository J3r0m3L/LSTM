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

reloaded_model = tf.keras.models.load_model('lstm_nlp_model')


# use model to predict
seed_text = lines[12343]

def generate_text_seq(model, tokenizer, text_seq_length, seed_text, n_words):
    text = []
    for _ in range(n_words):
        encoded = tokenizer.texts_to_sequences([seed_text])[0]
        encoded = pad_sequences([encoded], maxlen = text_seq_length, truncating = 'pre')

        y_predict = model.predict(encoded)
        y_predict = np.argmax(y_predict, axis=1)
        predicted_word = ''
        for word, index in tokenizer.word_index.items():
            if index == y_predict: 
                predicted_word = word
                break
                
        seed_text = seed_text + ' ' + predicted_word
        text.append(predicted_word)
    return ' '.join(text)

# if works should predict the next 10 words of the seed text
# Could not load symbol cublasGetSmCountTarget from cublas64_11.dll. Error code 127 really annoying error
print(generate_text_seq(reloaded_model, tokenizer, seq_length, seed_text, 10))
print('_____________________________________')
print(seed_text)
print('_____________________________________')
print(lines[12353])

# Works Great!