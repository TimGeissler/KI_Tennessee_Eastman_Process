import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random

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

    print('Look, i found a file: ' + filePath)

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



#falls geplottet werden soll
            #x = np.arange(480)
            #y = np.vstack(np.transpose(data[i]))
            #fig, ax = plt.subplots()
            #ax.plot(x, y)
            #plt.show()       

training_data = []

for i in range(22):
      #label
      oneHotVector = np.eye(22)[i, :]
      fault_data = np.transpose(data[i] / np.max(data))

      for k in range(480):
            training_data.append([fault_data[k], oneHotVector])

random.shuffle(training_data)

x_train = []
y_train = []

for testvalues, label in training_data:
      x_train.append(testvalues)
      y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)

# Erstellen des Modells bzw. Neuronalen Netzes 

# Sequential Model -> Feed-Forward Model 
model = tf.keras.models.Sequential()

#model.add(tf.keras.layers.Flatten()) # noch keine Ahnung was das bedeutet (Nacharbeiten)
model.add(tf.keras.layers.Dense(52, activation=tf.nn.relu))#, input_shape = x_train.shape[1:]))
# erste Schicht des NN mit 52 Neuronen und der Aktivierungsfunktion (z.B Sprungfunktion, Sigmoid-Funktion (Schwanenhals-Funktion))
#model.add(tf.keras.layers.Dense(52, activation = tf.nn.relu)) # Vorlesung
# zweite Schicht des NN mit 128 Neuronen und der Aktivierungsfunktion (z.B Sprungfunktion, Sigmoid-Funktion (Schwanenhals-Funktion))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
# Ausgangsschicht des NN mit 22 Neuronen, da es 21 Fälle gibt. ie Aktivierungsfunktion ist hier eine Wahrscheinlichkeitsverteilung
model.add(tf.keras.layers.Dense(22, activation = tf.nn.softmax))

model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

numberOfEpochs = 10

#print(x_train.shape)
#print(y_train.shape)
print(y_train[0])

#model.fit(x_train, y_train, epochs=numberOfEpochs)

#prediction = [] 
#for n in range(22):
#      print('now predicting: ' + str(n))
#      for o in range(480):
#            prediction.append(model.predict(np.transpose(dataTE[n])[m]))
#            print(np.argmax(model.predict(np.transpose(dataTE[n])[m])))




print('done without error lol how come')