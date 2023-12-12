#!/usr/bin/env python
# coding: utf-8

# In[125]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras import regularizers

# In[]:
def load_dat(path, input_end, sens, crop_start, crop_stop, flip=False):
    try:
        with open(path, 'r') as file:
            # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
            lines = file.read().splitlines()

            # Die Daten in eine Matrix umwandeln
            data = [list(map(float, line.split()[:input_end])) for line in lines]
            data = np.array(data)
            if flip:
                data = np.transpose(data)
            data = data[crop_start:crop_stop, sens]



        # Größe der neuen Matrix ausgeben
        new_size = (len(data), len(data[0]))
        print(f"Größe der Trainingsdaten Matrix: {new_size}")
        return data, new_size

    except FileNotFoundError:
        print(f'Die Datei {path} wurde nicht gefunden.')

    except Exception as e:
        print(f'Fehler beim Lesen der Datei: {str(e)}')


# In[]:
def build_model(input_shape, output_shape):
    # Sequential Model -> Feed-Forward Model
    model = Sequential()
    model.add(Dense(64, input_shape=(input_shape,), activation='relu'))#, kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dropout(0.3))
    model.add(Dense(32, activation='relu'))#, kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dropout(0.02))
    model.add(Dense(32, activation='relu'))#, kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dropout(0.1))
    model.add(Dense(output_shape, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# In[]:
data_matrix, new_matrix_size = load_dat(path='/Users/justinolivieri/Desktop/Künstliche Intelligenz/d00.dat',
                                        input_end=480,
                                        sens=[1, 2, 6, 12, 15, 18],
                                        crop_start=0,
                                        crop_stop=480,
                                        flip=True)


data_matrix_te0, new_matrix_size_te0 = load_dat(path='/Users/justinolivieri/Desktop/Künstliche Intelligenz/d00_te.dat',
                                                input_end=480,
                                                sens=[1, 2, 6, 12, 15, 18],
                                                crop_start=0,
                                                crop_stop=480)


data_matrix1, matrix_size1 = load_dat(path='/Users/justinolivieri/Desktop/Künstliche Intelligenz/d01.dat',
                                      input_end=480,
                                      sens=[1, 2, 6, 12, 15, 18],
                                      crop_start=0,
                                      crop_stop=480)


data_matrix_te1, new_matrix_size_te1 = load_dat(path='/Users/justinolivieri/Desktop/Künstliche Intelligenz/d01_te.dat',
                                                input_end=480,
                                                sens=[1, 2, 6, 12, 15, 18],
                                                crop_start=0,
                                                crop_stop=480)


data_matrix18, matrix_size18 = load_dat(path='/Users/justinolivieri/Desktop/Künstliche Intelligenz/d18.dat',
                                      input_end=480,
                                      sens=[1, 2, 6, 12, 15, 18],
                                      crop_start=0,
                                      crop_stop=480)


data_matrix_te18, new_matrix_size_te18 = load_dat(path='/Users/justinolivieri/Desktop/Künstliche Intelligenz/d18_te.dat',
                                                input_end=480,
                                                sens=[1, 2, 6, 12, 15, 18],
                                                crop_start=0,
                                                crop_stop=480)




# In[126]:


# In[]:
data_matrix_label = np.zeros(data_matrix.shape[0])
data_matrix_te0_label = np.zeros(data_matrix_te0.shape[0])

data_matrix1_label = np.zeros(data_matrix1.shape[0]) + 1
data_matrix_te1_label = np.zeros(data_matrix_te1.shape[0]) + 1

data_matrix18_label = np.zeros(data_matrix18.shape[0]) + 2
data_matrix_te18_label = np.zeros(data_matrix_te18.shape[0]) + 2


train_data = np.vstack((data_matrix, data_matrix1, data_matrix18))
train_data_label = np.hstack((data_matrix_label, data_matrix1_label, data_matrix18_label))
train_data_label = to_categorical(train_data_label, num_classes=3)

test_data = np.vstack((data_matrix_te0, data_matrix_te1, data_matrix_te18))
test_data_label = np.hstack((data_matrix_te0_label, data_matrix_te1_label, data_matrix_te18_label))
test_data_label = to_categorical(test_data_label, num_classes=3)



# In[131]:


# In[]:
x_train = train_data/np.linalg.norm(train_data, axis=0)
y_train = train_data_label

x_test = test_data/np.linalg.norm(test_data, axis=0)
y_test = test_data_label



# In[7]:
model = build_model(input_shape=6, output_shape=3)


model.fit(x_train, y_train,
          epochs=30,
          validation_data=(x_test, y_test),
          shuffle=True)


model.save('DFF_NN_V2.model')
new_model = tf.keras.models.load_model('DFF_NN_V2.model')


# In[9]:


predictions = new_model.predict([x_test])
print(predictions)


# In[ ]:




