import sys
import time

import pandas as pd
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import *
import matplotlib.pyplot as plt
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model
from imblearn.over_sampling import SMOTE
from time_logger import TimeLogger

import nltk
import numpy as np

# The maximum number of words to be used. (most frequent)
from keras.models import load_model

MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 32
# Stop words
stopwords_list = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
                  "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
                  "itself", "they", "them", "their", "theirs", "themselves", "which", "who", "whom", "these",
                  "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
                  "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as",
                  "until", "while", "of", "at", "by", "for", "with", "against", "into", "through", "during",
                  "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
                  "under", "again", "further", "then", "once", "here", "there", "when", "why", "how", "all", "any",
                  "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
                  "same", "so", "than", "too", "very", "s", "t", "don", "should", "now"]


def import_and_prepare(filepath):
    df = pd.read_csv(filepath, names=['sentence', 'operation'], sep=',', engine='python')
    # df = shuffle(df)
    sentences = df['sentence'].values
    y = df['operation'].values
    return df, sentences, y


def filter_stopwords(sentences, stopwords_list):
    stopwords_set = set(stopwords_list)
    filtered = []
    for sentence in sentences:
        tokenized_sentence = word_tokenize(sentence)
        filtered_sentence = []
        for w in tokenized_sentence:
            if w not in stopwords_set:
                filtered_sentence.append(w)
        filtered.append(filtered_sentence)
    return filtered


def detokenize(filtered_sentences):
    detokenized_sentences = []
    for sentence in filtered_sentences:
        detokenized_sentences.append(TreebankWordDetokenizer().detokenize(sentence))
    return detokenized_sentences


def plot_history(history):
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


def plot_label_distribution(dataframe):
    dataframe['operation'].value_counts().plot(kind="bar")


def init_tokenizer(MAX_NB_WORDS, dataframe):
    tokenizer = Tokenizer(MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(dataframe['filtered_sentence'].values)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    return tokenizer


def create_model(max_words, embedding_dimensions, X):
    model = Sequential()
    model.add(Embedding(max_words, embedding_dimensions, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=l2(0.01)))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def lstm_train(df, tokenizer, max_sequence_length, embedding_dimensions):
    X = tokenizer.texts_to_sequences(df['filtered_sentence'].values)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', X.shape)
    Y = pd.get_dummies(df['operation']).values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    # Oversampling the minority class
    smote = SMOTE('minority')
    X_train, Y_train = smote.fit_sample(X_train, Y_train)

    model = create_model(max_sequence_length, embedding_dimensions, X)
    epochs = 150
    batch_size = 100
    history = model.fit(X_train, Y_train,
                        epochs=epochs, batch_size=batch_size,
                        validation_split=0.1,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

    accr = model.evaluate(X_test, Y_test)
    print(model.summary())
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
    # plot_model(model, to_file='model.png')
    return model, history


def infer(sentence, tokenizer, model):
    sentence_as_array = [sentence]
    filtered_commands = filter_stopwords(sentence_as_array, stopwords_list)
    seq = tokenizer.texts_to_sequences(filtered_commands)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = model.predict(padded)
    return pred


def pre_initialize():
    df, sentences, y = import_and_prepare('data/dataset_new.txt')
    # df_temp, sentences_temp, y_temp = import_and_prepare('data/dataset_new.txt')
    plot_label_distribution(df)
    filtered_sentences = filter_stopwords(sentences, stopwords_list)
    detokenized_sentences = detokenize(filtered_sentences)
    df['filtered_sentence'] = detokenized_sentences
    tokenizer = init_tokenizer(MAX_NB_WORDS, df)
    return df, tokenizer


if __name__ == '__main__':
    # df, sentences, y = import_and_prepare('data/dataset.txt')
    # nltk.download()

    df, tokenizer = pre_initialize()
    # model, history = lstm_train(df, tokenizer, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)
    # model.save('lstm.h5')
    # plot_history(history)

    commands = pd.read_csv('data/commands.csv', encoding="utf-8")['commands']

    # ====== Test ========
    model = load_model('./lstm.h5')

    print('Size of model: ' + str(sys.getsizeof(model)) + ' bytes')

    dataframe = pd.DataFrame({'command': [], 'time': [], 'class': []})

    labels = ['Locate', 'Describe', 'No_Op']

    trials = 10

    for command in commands:
        _time = 0.0
        for i in range(trials):
            print("Command " + command)
            filtered_commands = filter_stopwords([command], stopwords_list)
            seq = tokenizer.texts_to_sequences(filtered_commands)
            padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
            start = time.time()
            pred = model.predict(padded)
            delta = time.time() - start
            _time = _time + delta
        dataframe = dataframe.append({'command': command, 'time': _time / trials, 'class': labels[np.argmax(pred)]}
                                     , ignore_index=True)

    dataframe.to_csv('predictions.csv', index=False)
    #
    # new_command = ['Can you describe this book']
    # filtered_commands = filter_stopwords(new_command, stopwords_list)
    # seq = tokenizer.texts_to_sequences(filtered_commands)
    # padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    # time_logger = TimeLogger('')
    # time_logger.start()
    # pred = model.predict(padded)
    # time_logger.end()
    #
    # labels = ['Locate', 'Describe', 'No_Op']
    # print("Predicted vector: ", pred, " Predicted Class: ", labels[np.argmax(pred)])
