import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import tensorflow as tf



X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])


model = Sequential()
model.add(Dense(4, input_dim=2, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

opt = tf.keras.optimizers.Adam(learning_rate=0.1)
model.compile(loss='binary_crossentropy', optimizer=opt)


history = model.fit(X, y, epochs=1000, verbose=0,batch_size=1,)




predictions = model.predict(X)

print(predictions)


plt.plot(history.history['loss'])
plt.title('Strata')
plt.xlabel('Epoki')
plt.ylabel('Strata')
plt.show()
