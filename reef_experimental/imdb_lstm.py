import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from pdb import set_trace as t

#top_words=5000
train_text = np.load('lstm_train_text.npy')
y_train = np.load('lstm_train_ground.npy')
test_text = np.load('lstm_test_text.npy')
y_test = np.load('lstm_test_ground.npy')
y_train[y_train == -1] = 0
y_test[y_test == -1] = 0
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_text)
tokenizer.fit_on_texts(test_text)
X_train = tokenizer.texts_to_sequences(train_text)
X_test = tokenizer.texts_to_sequences(test_text)
max_sentence_length=500
X_train = sequence.pad_sequences(X_train, maxlen=max_sentence_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_sentence_length)
embedding_vector_length = 32
vocab_size=len(tokenizer.word_index) + 1
model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_length, input_length=max_sentence_length))
model.add(LSTM(100))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
