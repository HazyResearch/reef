import numpy as np
import os 
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from pdb import set_trace as t

#top_words=5000

def lstm_glove(train_text, y_train, test_text, y_test, bs=64, n=3):
    #Label Processing
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0

    #FOR TRANSFER LEARNING
    train_idx = np.random.choice(np.shape(y_train)[0], 284) #imdb
    y_train = y_train[train_idx]
    train_text = train_text[train_idx]

    #Make Tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_text)
    tokenizer.fit_on_texts(test_text)
    X_train = tokenizer.texts_to_sequences(train_text)
    X_test = tokenizer.texts_to_sequences(test_text)

    #Preparing the Embedding Layer
    embeddings_index = {}
    f = open('/dfs/scratch0/paroma/data/glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_vector_length = 100
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_vector_length))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    max_sentence_length=500
    X_train = sequence.pad_sequences(X_train, maxlen=max_sentence_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_sentence_length)
    vocab_size=len(tokenizer.word_index) + 1

    #Model Architecture
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_vector_length, weights=[embedding_matrix], input_length=max_sentence_length, trainable=False))
    #model.add(Embedding(vocab_size, embedding_vector_length, input_length=max_sentence_length))
    model.add(LSTM(100))
    model.add(Dense(1, activation="sigmoid"))

    #Run the model!
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=n, batch_size=bs)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    y_pred = model.predict(X_test, batch_size=1)
    y_pred = np.array([x[0] for x in y_pred])
    return y_pred