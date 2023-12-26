
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import os


train_ds = keras.utils.image_dataset_from_directory(
  directory = "D:\Semester 7\ML\Project\dataset\Train",
  labels='inferred',
  label_mode = 'int',
  batch_size = 32,
  image_size = (416,416)
)
validation_ds = keras.utils.image_dataset_from_directory(
  directory = "D:\Semester 7\ML\Project\dataset\Validation",
  labels='inferred',
  label_mode = 'int',
  batch_size = 32,
  image_size = (416,416)
)

# Image Processing
def process(image,label):
  image = tf.cast(image/255. ,tf.float32)
  return image,label
train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)

model = Sequential()
model.add(Conv2D(16,(3,3),padding='valid',activation='relu',input_shape=(416,416,3)))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(32,(3,3),padding='valid',activation='relu',input_shape=(416,416,3)))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(64,(3,3),padding='valid',activation='relu',input_shape=(416,416,3)))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(128,(3,3),padding='valid',activation='relu',input_shape=(416,416,3)))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model_summary = model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history=model.fit(train_ds,epochs=6,validation_data=validation_ds)

save_directory = 'D:\Semester 7\ML\Project'
model_filename = 'HOI_model_416(1).h5'
model.save(os.path.join(save_directory, model_filename))

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'],color='red',label='train')
plt.plot(history.history['val_accuracy'],color='blue',label='validation')
plt.legend()
plt.show()