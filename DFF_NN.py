#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras import regularizers


# TensorFlow wird nicht benötigt, es sei denn, Sie verwenden es anderswo in Ihrem Code
# import tensorflow as tf

 #Öffnen der .dat-Datei im Lesemodus
file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d00.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()

        # Die Daten in eine Matrix umwandeln
        data_matrix = [list(map(float, line.split()[:480])) for line in lines]
        data_matrix = np.array(data_matrix)
        data_matrix = data_matrix[[1,2,6,12,15,18], 100:150]
        #data_matrix = np.transpose(data_matrix)
        

    # Größe der neuen Matrix ausgeben
    new_matrix_size = (len(data_matrix), len(data_matrix[0]))
    print(f"Größe der Trainingsdaten Matrix: {new_matrix_size}")

except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')

    
#Einlesen der Testdaten 
file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d00_te.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()

        # Die Daten in eine Matrix umwandeln
        data_matrix_te0 = [list(map(float, line.split()[:52])) for line in lines]
        data_matrix_te0 = np.array(data_matrix_te0)
        data_matrix_te0 = np.transpose(data_matrix_te0)
        data_matrix_te0 = data_matrix_te0[:, :480]
        data_matrix_te0 = data_matrix_te0[[1,2,6,12,15,18], 100:150]
        #data_matrix_te0 = np.transpose(data_matrix_te0)
        
        # Größe der neuen Matrix ausgeben
        new_matrix_size_te0 = (len(data_matrix_te0), len(data_matrix_te0[0]))
        print(f"Größe der Testdaten Matrix: {new_matrix_size_te0}")

except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
   print(f'Fehler beim Lesen der Datei: {str(e)}')


# In[2]:


file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d01.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()

        # Die Daten in eine Matrix umwandeln
        data_matrix1 = [list(map(float, line.split())) for line in lines]
        data_matrix1 = np.array(data_matrix1)
        data_matrix1 = np.transpose(data_matrix1)
        data_matrix1 = data_matrix1[:, :480]
        data_matrix1 = data_matrix1[[1,2,6,12,15,18], 100:150]
        #data_matrix1 = np.transpose(data_matrix1)

        #Größe der Matrix ausgeben
        matrix_size1 = (len(data_matrix1), len(data_matrix1[0]))
        print(f"Größe der Trainingsdatenmatrix: {matrix_size1}")

except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')

#Einlesen der Testdaten 
file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d01_te.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()
        
         # Die Daten in eine Matrix umwandeln
        data_matrix_te1 = [list(map(float, line.split()[:52])) for line in lines]
        data_matrix_te1 = np.array(data_matrix_te1)
        data_matrix_te1 = np.transpose(data_matrix_te1)
        data_matrix_te1 = data_matrix_te1[:, :480]
        data_matrix_te1 = data_matrix_te1[[1,2,6,12,15,18], 100:150]
        #data_matrix_te1 = np.transpose(data_matrix_te1)
        
        # Größe der neuen Matrix ausgeben
        new_matrix_size_te1 = (len(data_matrix_te1), len(data_matrix_te1[0]))
        print(f"Größe der Testdaten Matrix: {new_matrix_size_te1}")
    
except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


# In[ ]:





# In[ ]:





# In[3]:


# Erstellen des Modells bzw. Neuronalen Netzes 

# Sequential Model -> Feed-Forward Model 
model = Sequential()
model.add(Dense(16, input_shape=(50,), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(0.5))
model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(0.5))
model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(0.5))
model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(0.5))
model.add(Dense(2, activation='softmax'))


model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])


# In[5]:


x_train = data_matrix/np.linalg.norm(data_matrix, axis=0)
y_train = np.array([1, 0, 0, 0, 0, 0])

x_test = data_matrix_te0/np.linalg.norm(data_matrix_te0, axis=0) #saklierung keine normierung L2 Normierung 
y_test = y_train

y_train_one_hot = to_categorical(y_train, num_classes=2)
y_test_one_hot = to_categorical(y_test, num_classes=2)


# In[ ]:





# In[6]:


x_train1 = data_matrix1/np.linalg.norm(data_matrix1, axis=0)

y_train1 = np.array([0, 0, 0, 0, 0, 1])

x_test1 = data_matrix_te1/np.linalg.norm(data_matrix_te1, axis=0)
y_test1 = y_train1

y_train_one_hot1 = to_categorical(y_train1, num_classes=2)
y_test_one_hot1 = to_categorical(y_test1, num_classes=2)


# In[7]:


model.fit(x_train1, y_train_one_hot1, epochs=1, validation_data=(x_test1, y_test_one_hot1))
model.fit(x_train, y_train_one_hot, epochs=1, validation_data=(x_test, y_test_one_hot))
model.fit(x_train1, y_train_one_hot1, epochs=1, validation_data=(x_test1, y_test_one_hot1))
model.fit(x_train, y_train_one_hot, epochs=1, validation_data=(x_test, y_test_one_hot))
model.fit(x_train1, y_train_one_hot1, epochs=1, validation_data=(x_test1, y_test_one_hot1))
model.fit(x_train, y_train_one_hot, epochs=1, validation_data=(x_test, y_test_one_hot))
model.fit(x_train, y_train_one_hot, epochs=1, validation_data=(x_test, y_test_one_hot))
model.fit(x_train, y_train_one_hot, epochs=1, validation_data=(x_test, y_test_one_hot))
model.fit(x_train, y_train_one_hot, epochs=1, validation_data=(x_test, y_test_one_hot))
model.fit(x_train1, y_train_one_hot1, epochs=1, validation_data=(x_test1, y_test_one_hot1))
model.fit(x_train1, y_train_one_hot1, epochs=1, validation_data=(x_test1, y_test_one_hot1))
model.fit(x_train1, y_train_one_hot1, epochs=1, validation_data=(x_test1, y_test_one_hot1))
model.fit(x_train, y_train_one_hot, epochs=1, validation_data=(x_test, y_test_one_hot))
model.fit(x_train1, y_train_one_hot1, epochs=1, validation_data=(x_test1, y_test_one_hot1))
model.fit(x_train, y_train_one_hot, epochs=1, validation_data=(x_test, y_test_one_hot))
model.fit(x_train1, y_train_one_hot1, epochs=1, validation_data=(x_test1, y_test_one_hot1))
model.fit(x_train, y_train_one_hot, epochs=1, validation_data=(x_test, y_test_one_hot))
model.fit(x_train, y_train_one_hot, epochs=1, validation_data=(x_test, y_test_one_hot))
model.fit(x_train, y_train_one_hot, epochs=1, validation_data=(x_test, y_test_one_hot))
model.fit(x_train, y_train_one_hot, epochs=1, validation_data=(x_test, y_test_one_hot))
model.fit(x_train1, y_train_one_hot1, epochs=1, validation_data=(x_test1, y_test_one_hot1))
model.fit(x_train1, y_train_one_hot1, epochs=1, validation_data=(x_test1, y_test_one_hot1))


# In[ ]:





# In[ ]:





# In[8]:


model.save('DFF_NN_V2.model')
new_model = tf.keras.models.load_model('DFF_NN_V2.model')


# In[9]:


predictions = new_model.predict([x_test])
print(predictions)


# In[10]:


predictions = new_model.predict([x_test1])
print(predictions)


# In[ ]:





# In[ ]:





# In[ ]:




