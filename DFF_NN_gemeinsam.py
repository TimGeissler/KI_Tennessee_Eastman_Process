import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random

from sklearn.metrics import confusion_matrix
import seaborn as sns

####################################################################################
# Trainingsparameter

test_Number = 25
numberOfEpochs = 200
batch_Size = 10
learningRate = 0.1

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

####################################################################################
# durchmischen
            
def shuffleWithBatchSize(pList, pBatchSize):
    # Autor: TG, 19.12.2023
    # Die Funktion nimmt eine übergebene Liste und gibt eine durchmischte Liste zurück.
    # Bei der Durchmischung werden immer <pBatchSize> ursprüngliche Werte zu einem Batch zusammengefasst.
    # Im Anschluss werden diese Batches vermischt und anschließend zu einer Liste der ursprünglichen Form zusammengefasst.
    # Falls eine BatchSize gewählt wurde, die kein Teiler der Listengröße ist, so wird ein (bzw. der letzte) Batch entsprechend kleiner, da er den Rest der Werte enthält.


    # Stelle sicher, dass die Batch-Größe kleiner oder gleich der Länge der Liste ist
    batch_size_true = min(pBatchSize, len(pList))
    
    # Bestimme die Anzahl der Batches
    # Falls Batchgröße kein Teiler der Listenlänge ist, addiere 1 um letzte Werte in kleinerem Batch zu sammeln
    num_batches = len(pList) // batch_size_true
    if (num_batches*batch_size_true < len(pList)):
        num_batches += 1

    # Erzeuge Batches
    batches = []
    for i in range(num_batches):
        batch = []
        for k in range(batch_size_true):
            index = i*batch_size_true + k
            if (index < len(pList)):
                batch.append(pList[index])
        batches.append(batch)

    # Durchmische Batches
    random.shuffle(batches)

    # Kombiniere die durchmischten Batches zu einer durchmischten Liste
    shuffled_list = []
    for batch in batches:
        for item in batch:
            shuffled_list.append(item)

    return shuffled_list

training_data = shuffleWithBatchSize(training_data, batch_Size) # anpassen mit tf-Funktion mit batch-size etc.

####################################################################################

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

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate),
             loss='categorical_crossentropy',
             metrics=['accuracy'])



#print(x_train.shape)
#print(y_train.shape)
#print(y_train[0])

model.fit(x_train, y_train, epochs=numberOfEpochs,validation_data=(x_test, y_test))

predictions = model.predict([x_test])
#print(predictions.shape)

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
#plt.show()
path_scatter = 'Plots\\ScatterPlot\\' + str(test_Number) + '.png'
if not os.path.exists(path_scatter):
    plt.savefig(path_scatter)
else:
    print('WARNING: file already exists and will not be overwritten. Please change the testNumber')

#for i in range(22):
    #print(np.argmax(predictions[i]))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test_vector, predictions_vector)
#print('Confusion Matrix:')
#print(conf_matrix)

#Heatmap 
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Vorhergesagt')
plt.ylabel('Tatsächlich')
plt.title('Confusion Matrix')
#plt.show()
path_heatmap = 'Plots\\Heatmap (ConfusionMatrix)\\' + str(test_Number) + '.png'
if not os.path.exists(path_heatmap):
    plt.savefig(path_heatmap)
else:
    print('WARNING: file already exists and will not be overwritten. Please change the testNumber')




print('done without error lol how come')