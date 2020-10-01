from keras.preprocessing import sequence 
from keras.models import Sequential 
from keras.layers import Dense, Embedding 
from keras.layers import LSTM 
from keras.datasets import imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 2000)

x_train = sequence.pad_sequences(x_train, maxlen=80) 
x_test = sequence.pad_sequences(x_test, maxlen=80)

model = Sequential() 
model.add(Embedding(2000, 128)) 
model.add(LSTM(128, dropout = 0.2, recurrent_dropout = 0.2)) 
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

with tf.device('/device:GPU:0'):
  model.fit(
    x_train, y_train, 
    batch_size = 128, 
    epochs = 10, 
    validation_data = (x_test, y_test)
  )
  
score, acc = model.evaluate(x_test, y_test, batch_size = 32) 
   
print('Test score:', score) 
print('Test accuracy:', acc)
