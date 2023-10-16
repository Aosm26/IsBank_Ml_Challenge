import pandas as pd
import tensorflow as tf
import numpy as np
import math
from sklearn.impute import SimpleImputer
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data=pd.read_csv('bos.csv')

xdata = data.drop(columns=['menus','id','month','carrier','devicebrand','n_seconds_1','n_seconds_2','n_seconds_3','menus_1','menus_2','menus_3','menus_4','menus_5','menus_6','menus_7','menus_8','menus_9'])
sutunlar=['menus_2']
ydata = data[sutunlar].values

x=(xdata-np.min(xdata))/(np.max(xdata)-np.min(xdata))
y=(ydata-np.min(ydata))/(np.max(ydata)-np.min(ydata))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.00001,random_state=1)

model = tf.keras.models.Sequential()
print(x.shape)
print(y.shape)
model.add(tf.keras.layers.Dense(256, input_shape=(50,), activation=None))
model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(64, activation='relu'))
# model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='relu'))
# model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(4, activation='relu'))
# model.add(tf.keras.layers.Dense(2, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.FalseNegatives()])

# model.evaluate(x_test, y_test)

model.fit(x_train, y_train, epochs=2)

# model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)
test_loss = model.evaluate(x_test, y_test)
print(f'Test verileri üzerinde kayıp ve doğruluk: {test_loss}')

y_pred = model.predict(x_test)
print(y_pred.tolist())

# model.add(layers.Dense(256, input_shape=(50,), activation='relu'))

# # Gizli katmanlar
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dense(64, activation='relu'))

# # Çıkış katmanı (Sınıflandırma probleminde sıklıkla 'softmax' kullanılır)
# model.add(layers.Dense(1, activation='softmax'))

# # Modeli derle
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



# print(y_train.shape)
# print(x_train.shape)
# model.fit(x_train, y_train, epochs=500)

# model = models.Sequential()

# # Giriş katmanı
# model.add(layers.Dense(256, input_shape=(49,), activation='relu'))

# # Gizli katmanlar
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dense(64, activation='relu'))

# # Çıkış katmanı (Sınıflandırma probleminde sıklıkla 'softmax' kullanılır)
# model.add(layers.Dense(3, activation='softmax'))

# # Modeli derle
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # model.evaluate(x_test, y_test)

# model.fit(x_train, y_train, epochs=1000)