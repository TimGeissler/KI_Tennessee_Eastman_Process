import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import os

# Training auf GPU verlagern
#physical_devices = tf.config.experimental.list_physical_devices
#print(physical_devices)

#RELPATH = 'D:\\Tim\\Programmierung\\Playground\\Daten\\'
RELPATH = 'Dataset\\'

data = []
dataTE = []
index = 0
indexTE = 0

# Dateien einlesen
for datei in os.listdir(RELPATH):
    filePath = RELPATH + datei

    #print('Look, i found a file: ' + filePath)

    if 'te' in datei:
        with open(filePath, 'r') as file:
                #Datei einlesen
                lines = file.read().splitlines()
                dataTE.append([list(map(float, line.split()[:52])) for line in lines])
                
                dataTE[indexTE] = np.transpose(dataTE[indexTE])
                dataTE[indexTE] = dataTE[indexTE][:, :480]

                size = (len(dataTE[indexTE]), len(dataTE[indexTE][0]))
                #print(f'Größe der Matrix aus {datei}: {size}')
                indexTE += 1
    else:
        with open(filePath, 'r') as file:
                lines = file.read().splitlines()
                data.append([list(map(float, line.split()[:480])) for line in lines])

                size = (len(data[index]), len(data[index][0]))
                if (size[0] > 52):
                      data[index] = np.transpose(data[index])
                #print(f'Größe der Matrix aus {datei}: {size}')
                index += 1

# Erstellen des Modells bzw. Neuronalen Netzes 

# Sequential Model -> Feed-Forward Model 
model = tf.keras.models.Sequential()

#model.add(tf.keras.layers.Flatten()) # noch keine Ahnung was das bedeutet (Nacharbeiten)

# erste Schicht des NN mit 52 Neuronen und der Aktivierungsfunktion (z.B Sprungfunktion, Sigmoid-Funktion (Schwanenhals-Funktion))
model.add(tf.keras.layers.Dense(64, activation = tf.nn.relu)) # Vorlesung
# zweite Schicht des NN mit 128 Neuronen und der Aktivierungsfunktion (z.B Sprungfunktion, Sigmoid-Funktion (Schwanenhals-Funktion))
model.add(tf.keras.layers.Dense(256, activation = tf.nn.relu))
# Ausgangsschicht des NN mit 22 Neuronen, da es 21 Fälle gibt. ie Aktivierungsfunktion ist hier eine Wahrscheinlichkeitsverteilung
model.add(tf.keras.layers.Dense(22,  activation = tf.nn.relu)) #stimmt so noch nicht mit 22 Fällen 
model.add(tf.keras.layers.Dense(22, activation = tf.nn.softmax))

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

#predictions = []

for m in range(10):
      for i in range(22):
            #falls geplottet werden soll
            #x = np.arange(480)
            #y = np.vstack(np.transpose(data[i]))
            #fig, ax = plt.subplots()
            #ax.plot(x, y)
            #plt.show()
            x_train = np.transpose(data[i-1] / np.max(data[i-1]))
            #print(x_train.shape)
            y_train = np.full(480, i)
            #y_train = np.array(1)
            numberOfEpochs = 10
            model.fit(x_train, y_train, epochs=numberOfEpochs)

prediction = [] 
for n in range(22):
      print('now predicting: ' + str(n))
      for o in range(480):
            prediction.append(model.predict(np.transpose(dataTE[n])[m]))
            print(np.argmax(model.predict(np.transpose(dataTE[n])[m])))


#x_train = np.array(data / np.max(data))
#print(x_train.shape)
#y_train = np.arange(22)
#model.fit(x_train, y_train, epochs=1000) # läuft einigermaßen


print('done without error lol how come')