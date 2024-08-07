

import tensorflow as tf
import numpy as np

layer_1 = tf.keras.layers.Dense(units=100, activation='sigmoid')
layer_2 = tf.keras.layers.Dense(units=25, activation='sigmoid')
layer_3 = tf.keras.layers.Dense(units=1, activation='sigmoid')

model=tf.keras.models.Sequential([layer_1,layer_2,layer_3])

x=np.array([[200,17],[120,5],[425,20],[212,18],[50,3]])
y=np.array([1,0,0,1,0])

model.compile(
              loss='binary_crossentropy'
              )

model.fit(x,y,epochs=500)
x_new=([[200,15],[50,1]])
out=model.predict(x_new)
print(out)

