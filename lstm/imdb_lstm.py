import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

#top_words=5000

def lstm_simple(train_text, y_train, test_text, y_test, bs=64, n=3):
    #Label Processing
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0

    #Make Tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_text)
    tokenizer.fit_on_texts(test_text)
    X_train = tokenizer.texts_to_sequences(train_text)
    X_test = tokenizer.texts_to_sequences(test_text)

    #Make embedding 
    max_sentence_length=500
    X_train = sequence.pad_sequences(X_train, maxlen=max_sentence_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_sentence_length)
    embedding_vector_length = 32
    vocab_size=len(tokenizer.word_index) + 1

    #Model Architecture
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_vector_length, input_length=max_sentence_length))
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
