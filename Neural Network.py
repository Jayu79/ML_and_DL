from keras.datasets import mnist
import tensorflow as tf
mnist = mnist.load_data()
(x_train,y_train),(x_test,y_test) = mnist
x_train = x_train/255.0
x_test = x_test/255.0
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape =(28,28)),
    tf.keras.layers.Dense(128,activation = 'relu'),
    tf.keras.layers.Dense(10,activation = 'softmax')
])
model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrices = ['accuracy'])
model.fit(x_train,y_train,epochs = 10)
model.evaluate(x_test,y_test)