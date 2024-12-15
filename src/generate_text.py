# %%
import random
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

# %%
data_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'

data_path = tf.keras.utils.get_file("dataset.txt", data_URL)

text = open(data_path, "rb").read().decode("utf-8").lower()

text = text[300000:800000]

# %%
vocab = sorted(set(text))

char_to_index = {c: i for i, c in enumerate(vocab)}
index_to_char = {i:c for i, c in enumerate(vocab)}

# %%
sequenceSize = 40
stepSize = 3

# %%
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length, temp):
    generated = ""
    starting_index = random.randint(0, len(text) - sequenceSize - 1)
    sentence = text[starting_index : starting_index + sequenceSize]
    generated += sentence
    
    for i in range(length):
        x = np.zeros((1, sequenceSize, len(vocab)))
        for t, character in enumerate(sentence):
            x[0, t, char_to_index[character]] = 1
        preds = model.predict(x, verbose=0)[0]
        nextCharIndex = sample(preds)
        next_char = index_to_char[nextCharIndex]
        generated += next_char
        sentence = sentence[1:] + next_char
    return generated
        

# %%
model = tf.keras.models.load_model("../model/text_generator.h5")

print(generate_text(300, 1))