## Code to fit character level LSTM(s) for language modeling
## Define two LSTM models (simple and deep)
## After each epoch, the model outputs some example generated text
##
## Import packages
import numpy as np
from urllib import request
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import LambdaCallback, ModelCheckpoint
import random
import sys


# we are going to use moby dick as our text
# change this url to any text file you want to use
url = 'http://www.gutenberg.org/files/2701/2701-0.txt'
response = request.urlopen(url)
# raw_txt should be string with the raw text you are using
raw_txt = response.read().decode('utf8').replace('\r', '')


# get size of data and vocbulary
chars = list(set(raw_txt))
data_size, vocab_size = len(raw_txt), len(chars)
print('Data has %d characters, with %d unique' % (data_size, vocab_size))


# create dictionary to map characters to index and vice versa
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for ch,i in char_to_ix.items() }


# organize into sequences of characters, encode into integer encoding
# choose length of sequences to use for model fitting, longer == more computationally intensive
length = 40
sequences = list()
for i in range(length, len(raw_txt)):
    # select sequence of tokens
    seq = raw_txt[i-length:i]
    encoded_seq = [char_to_ix[char] for char in seq]
    # temporary store
    sequences.append(encoded_seq)
print('Total Sequences: %d' % len(sequences))


# separate into input and output, convert to one-hot encoding
sequences = np.array(sequences)
X, y = sequences[:-1,:], sequences[1:,-1]
X = to_categorical(X, num_classes=vocab_size)
y = to_categorical(y, num_classes=vocab_size)


# define simple LSTM model
'''
model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# compile model
'''

# define the deep LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



# some code borrowed from here: https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
# prints sample text after each epoch
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(raw_txt) - length - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = raw_txt[start_index: start_index + length]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, length, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_to_ix[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = ix_to_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()


print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
# checkpoint criteria to save
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')


# fit model(s), you probably only want to run one at a time!
model.fit(X, y, epochs=500, batch_size=128, verbose=1, callbacks=[print_callback, checkpoint])

# save the model to file
model.save('model.h5')

