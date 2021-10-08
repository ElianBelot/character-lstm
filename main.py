# ===============[ IMPORTS ]===============
from keras.layers import LSTM, Dense, BatchNormalization, Dropout
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.models import Sequential

import matplotlib.pyplot as plt
import numpy as np
import json


# ===============[ HYPERPARAMETERS ]===============
EPOCHS = 100
BATCH_SIZE = 32
CHARACTER_THRESHOLD = 12        # All words containing characters appearing less than this threshold in the dataset will be deleted

GENERATE_ON_EPOCH_END = True    # Whether to generate samples when epochs are printed
GENERATE_EVERY = 10             # How many epochs needed for the samples to be generated
SAMPLES_AMOUNT = 5              # How many samples per generation

DATA_PATH = 'data/flowers_list.json'
CHECKPOINTS_PATH = 'checkpoints/weights-{epoch:02d}-{loss:.2f}.hdf5'


# ===============[ DATA PRE-PROCESSING ]===============
data = json.load(open(DATA_PATH, 'r'))
words = [word.lower().strip() + '.' for word in data]   # Converts all words to lowercase and add the stop word . at the end of them
data_string = ''.join(words)    # Creates a single string containing all the words


# ===============[ FILTERING RARE CHARACTERS ]===============
characters_count = {c: data_string.count(c) for c in set(data_string)}  # Dictionary that matches every character (key) with its count in the dataset (value)
rare_characters = [char for char, count in characters_count.items() if count < CHARACTER_THRESHOLD]  # Isolates all characters that appear less than the threshold
words = list(filter(lambda word: not any(char in rare_characters for char in word), words))  # Deletes every word that contains a rare character


# ===============[ VOCABULARY CREATION ]===============
vocabulary = sorted(set(''.join(words)))    # All common characters found in the dataset
char_to_index = {char: index for index, char in enumerate(vocabulary)}  # Matches each character to a unique index
index_to_char = {index: char for index, char in enumerate(vocabulary)}  # Matches each index to a unique character


# ===============[ DATASET INITIALIZATION ]===============
m = len(words)
max_length = len(max(words, key=len))   # Length of the longest word in the dataset
vocabulary_size = len(vocabulary)       # How many different characters are used

X = np.zeros((m, max_length, vocabulary_size))
Y = np.zeros((m, max_length, vocabulary_size))


# ===============[ ENCODING / DECODING ]===============
# Converts a string into a square matrix where every row is the one-hot representation of the character
def matrix_to_string(matrix):
    string = []

    # For every row (one-hot vector representing a character)
    for i in range(matrix.shape[0]):
        # Stop reading if the row is empty
        if 1 not in matrix[i]:
            break

        # Matches the index of the 1 in the row to the corresponding character
        string.append(index_to_char[matrix[i].tolist().index(1)])

    return ''.join(string)


def string_to_matrix(string):
    matrix = np.zeros((max_length, vocabulary_size))

    # For every characterof the string
    for i in range(len(string)):
        # Put a 1 at the index corresponding to the character
        matrix[i, char_to_index[string[i]]] = 1

    return matrix


# ===============[ DATASET POPULATION (ONE-HOT ENCODING) ]===============
# Populates every layer of the tensor with the matrix representation of each word
for i in range(m):
    X[i] = string_to_matrix(words[i])
    Y[i] = string_to_matrix(words[i][1:])


# ===============[ GENERATING NEW WORDS ]===============
def generate_word(model):
    word = []
    word_matrix = np.zeros((1, max_length, vocabulary_size))

    for i in range(max_length):
        # Model's prediction of the probability of each word based on the previous characters
        probs = list(model.predict(word_matrix)[0, i])
        probs /= np.sum(probs)

        # Sample a character based on its probability
        index = np.random.choice(range(vocabulary_size), p=probs)

        if i == max_length - 2:
            break
        else:
            character = index_to_char[index]

        if character == '.':
            break

        word.append(character)
        word_matrix[0, i + 1, index] = 1

    return ''.join(word).capitalize()


# ===============[ ON EPOCH END CALLBACK ]===============
def print_samples(epoch, logs):
    if GENERATE_ON_EPOCH_END and epoch % GENERATE_EVERY == 0:
        print('\n' + f'[EPOCH: {epoch} - LOSS: {logs["loss"]:.4f}]'.center(50, '='))
        for i in range(SAMPLES_AMOUNT):
            print(generate_word(model))


# ===============[ CALLBACKS ]===============
checkpoint_callback = ModelCheckpoint(CHECKPOINTS_PATH, verbose=0, monitor='loss', mode='min', save_best_only=True, save_weights_only=True)
generator_callback = LambdaCallback(on_epoch_end=print_samples)
callbacks = [checkpoint_callback, generator_callback]


# ===============[ MODEL ]===============
model = Sequential()

model.add(LSTM(256, input_shape=(max_length, vocabulary_size), return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(LSTM(128, input_shape=(max_length, vocabulary_size), return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(128, activation='tanh'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(64, activation='tanh'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(vocabulary_size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')


# ===============[ TRAINING ]===============
history = model.fit(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=0)


# ===============[ PLOTTING ]===============
plt.plot(history.history['loss'])
plt.title('Model training performance')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig(f'model_loss_{EPOCHS}_epochs.png')
plt.show()
