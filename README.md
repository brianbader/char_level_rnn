# char_level_rnn
Character level LSTM deep learning model for language modeling

## Description
This code downloads a text from project Gutenberg (currently Moby Dick) and uses it as the training set for a character level RNN (LSTM) model.

It first creates a vocabulary/index of characters using dictionaries, then creates vectors of one-hot encoded sequences of a fixed length, rolling the sequences one character at a time. So there will be len(text) - len(sequences) + 1 vectors as the input, with vectors of length equal to the vocabulary size.

Currently, two LSTM models are fit -- first a 'simple' LSTM, with a single layer of 128 units and dropout of 0.2, and the second a deeper LSTM with two layers of 256 and 64, with dropouts of 0.2 in between. 

After each epoch, some sample text is output from the model using various levels of the <a href="https://cs.stackexchange.com/questions/79241/what-is-temperature-in-lstm-and-neural-networks-generally/79242#79242">temperature</a> hyperparameter to inspect the reasonableness of the model predictions.
