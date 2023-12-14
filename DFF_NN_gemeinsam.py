import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random

####################################################################################

# Funktion zum Einlesen einer Datei
def load_dat(path, input_end, crop_start, crop_stop):
    try:
        with open(path, 'r') as file:
            # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
            lines = file.read().splitlines()

            # Die Daten in eine Matrix umwandeln
            data = [list(map(float, line.split()[:input_end])) for line in lines]
            data = np.array(data)

            # Format ermitteln und ggf. transponieren
            size = (len(data), len(data[0]))
            if (size[0] > 52):
                      data = np.transpose(data)
            
            data = data[crop_start:crop_stop]

        # Größe der neuen Matrix ausgeben
        #new_size = (len(data), len(data[0]))
        #print(f"Größe der Trainingsdaten Matrix: {new_size}")
        return data

    except FileNotFoundError:
        print(f'Die Datei {path} wurde nicht gefunden.')

    except Exception as e:
        print(f'Fehler beim Lesen der Datei: {str(e)}')


#RELPATH = 'C:\\Path\\if\\files\\are\\on\\your\\PC\\'
RELPATH = 'Dataset\\'

data_raw = []
dataTE_raw = []

# Dateien einlesen
for datei in sorted(os.listdir(RELPATH)):
    filePath = RELPATH + datei

    #print('Look, i found a file: ' + filePath)

    if 'te' in datei:
        data_matrix = load_dat(filePath, 480, 0, 480)
        dataTE_raw.append(data_matrix)
    else:
        data_matrix = load_dat(filePath, 480, 0, 480)
        data_raw.append(data_matrix)


data_raw = np.array(data_raw)
dataTE_raw = np.array(dataTE_raw)

####################################################################################

# normieren
data_norm = []
dataTE_norm = []

for i in range(22):
    array2ndD = []
    array2ndDTE = []

    for k in range(52):
        normOfSensorData = np.linalg.norm(data_raw[i][k])
        normOfSensorDataTE = np.linalg.norm(dataTE_raw[i][k])
        meanOfSensorData = np.mean(data_raw[i][k])
        meanOfSensorDataTE = np.mean(dataTE_raw[i][k])
        
        array3rdD = []
        array3rdDTE = []
            
        for m in range(480):
            array3rdD.append((data_raw[i][k][m] - meanOfSensorData)/normOfSensorData)
            array3rdDTE.append((dataTE_raw[i][k][m] - meanOfSensorDataTE)/normOfSensorDataTE) ### clean up

        array2ndD.append(array3rdD)
        array2ndDTE.append(array3rdDTE)

    data_norm.append(array2ndD)
    dataTE_norm.append(array2ndDTE)

data_norm = np.array(data_norm)
dataTE_norm = np.array(dataTE_norm)


# Debug 
# print(np.max(data_norm))
# print(np.min(data_norm))
#print(data_norm)
#print(dataTE_norm.shape)

#print(data_raw[1][1].shape)
#print(data_raw[1][1])
#print(np.linalg.norm(data_raw[1], axis=1))

####################################################################################

# aus normierter Matrix Trainigsdaten erstellen

training_data = []
test_data = []

for i in range(22):
      # hier label erzeugen -> one-hot vector
      oneHotVector = np.eye(22)[i, :]
      fault_data = np.transpose(data_norm[i])
      fault_dataTE = np.transpose(dataTE_norm[i])

      for k in range(480):
            training_data.append([fault_data[k], oneHotVector])
            test_data.append([fault_dataTE[k], oneHotVector])

# durchmischen

random.shuffle(training_data) # anpassen mit tf-Funktion mit batch-size etc.

### Training data
x_train = []
y_train = []

for testvalues, label in training_data:
      x_train.append(testvalues)
      y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)


#### Test data
x_test = []
y_test = []

for testvalues, label in test_data:
      x_test.append(testvalues)
      y_test.append(label)

x_test = np.array(x_test)
y_test = np.array(y_test)

####################################################################################
# Erstellen des Modells bzw. Neuronalen Netzes 

# Sequential Model -> Feed-Forward Model 
model = tf.keras.models.Sequential()

#model.add(tf.keras.layers.Flatten()) # noch keine Ahnung was das bedeutet (Nacharbeiten)
model.add(tf.keras.layers.Dense(52, activation=tf.nn.relu))
#model.add(tf.keras.layers.Dropout(0.2))#, input_shape = x_train.shape[1:]))
# erste Schicht des NN mit 52 Neuronen und der Aktivierungsfunktion (z.B Sprungfunktion, Sigmoid-Funktion (Schwanenhals-Funktion))
#model.add(tf.keras.layers.Dense(52, activation = tf.nn.relu)) # Vorlesung
# zweite Schicht des NN mit 128 Neuronen und der Aktivierungsfunktion (z.B Sprungfunktion, Sigmoid-Funktion (Schwanenhals-Funktion))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
#model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
#model.add(tf.keras.layers.Dropout(0.2))
# Ausgangsschicht des NN mit 22 Neuronen, da es 21 Fälle gibt. ie Aktivierungsfunktion ist hier eine Wahrscheinlichkeitsverteilung
model.add(tf.keras.layers.Dense(22, activation = tf.nn.softmax))

model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

numberOfEpochs = 100

#print(x_train.shape)
#print(y_train.shape)
#print(y_train[0])

model.fit(x_train, y_train, epochs=numberOfEpochs,validation_data=(x_test, y_test))

predictions = model.predict([x_test])
print(predictions.shape)

predictions_vector = []
y_test_vector = []
for i in range(10560):
    predictions_vector.append(np.argmax(predictions[i]))
    y_test_vector.append(np.argmax(y_test[i]))

predictions_vector = np.array(predictions_vector)
y_test_vector = np.array(y_test_vector)

plt.scatter(range(10560),y_test_vector,c='g')
plt.scatter(range(10560),predictions_vector,c='r')
#plt.scatter(range(10560),y_test_vector,c='g')
plt.show()
#for i in range(22):
    #print(np.argmax(predictions[i]))





print('done without error lol how come')