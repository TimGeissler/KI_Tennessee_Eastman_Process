import tensorflow as tf
import numpy as np
import os

RELPATH = 'D:\\Tim\\Programmierung\\Playground\\Daten\\'

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
                print(f'Größe der Matrix aus {datei}: {size}')
                indexTE += 1
    else:
        with open(filePath, 'r') as file:
                lines = file.read().splitlines()
                data.append([list(map(float, line.split()[:480])) for line in lines])

                size = (len(data[index]), len(data[index][0]))
                if (size[0] > 52):
                      data[index] = np.transpose(data[index])
                print(f'Größe der Matrix aus {datei}: {size}')
                index += 1

#Debug
#print(index)
#print(indexTE)
#



# für jeden Fehler (alle 22 Dateien) trainieren
# 



# Erstellen des Modells bzw. Neuronalen Netzes 

# Sequential Model -> Feed-Forward Model 
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten()) # noch keine Ahnung was das bedeutet (Nacharbeiten)

# erste Schicht des NN mit 52 Neuronen und der Aktivierungsfunktion (z.B Sprungfunktion, Sigmoid-Funktion (Schwanenhals-Funktion))
model.add(tf.keras.layers.Dense(52, activation = tf.nn.relu)) # Vorlesung
# zweite Schicht des NN mit 128 Neuronen und der Aktivierungsfunktion (z.B Sprungfunktion, Sigmoid-Funktion (Schwanenhals-Funktion))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
# Ausgangsschicht des NN mit 21 Neuronen, da es 21 Fälle gibt. ie Aktivierungsfunktion ist hier eine Wahrscheinlichkeitsverteilung
model.add(tf.keras.layers.Dense(21,  activation = tf.nn.softmax)) #stimmt so noch nicht mit 21 Fällen 

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

for faulttype in data:
      x_train = np.array(faulttype)
      #print(x_train.shape)
      #x_train_vector = np.array(x_train[:,0])
      #print(x_train_vector.shape)
      y_train = np.random.randint(21, size=(x_train.shape[0]))

      model.fit(x_train, y_train, epochs=100) # Hier scheint es ein problem damit zu geben das lable0 


print('done without error lol how come')