import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding


reviews = ['good food',
        'amazing restaurant',
        'too good',
        'just loved it!',
        'will go again',
        'horrible food',
        'never go there',
        'poor service',
        'poor quality',
        'needs improvement']

labels = np.array([1,1,1,1,1,0,0,0,0,0])

vocab_size = 50
encoded_reviews = [one_hot(d, vocab_size) for d in reviews]
max_length = 4
padded_reviews = pad_sequences(encoded_reviews, maxlen=max_length, padding='post')

embeded_vector_size = 5

model = Sequential()
model.add(Embedding(vocab_size, embeded_vector_size, input_length=max_length,name="embedding"))
model.add(LSTM(32))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

X = padded_reviews
y = labels

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

model.fit(X, y, epochs=5, verbose=1)