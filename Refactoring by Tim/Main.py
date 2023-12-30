import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random

from sklearn.metrics import confusion_matrix
import seaborn as sns

import ReadData as read
import EditData as edit

####################################################################################
# Organisatorisches
#RELPATH = 'C:\\Path\\if\\files\\are\\on\\your\\PC\\'
RELPATH = 'Dataset\\'


# Trainingsparameter

test_Number = 25
numberOfEpochs = 200
batch_Size = 10
learningRate = 0.001

####################################################################################

# Funktion zum Einlesen einer Datei

data_raw, dataTE_raw = read.loadDataFromDirectory(RELPATH, 480, 0, 480)


####################################################################################

# normieren
data_norm = edit.datensatzNormieren(data_raw)
dataTE_norm = edit.datensatzNormieren(dataTE_raw)



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

training_data = edit.shuffleWithBatchSize(training_data, batch_Size) # anpassen mit tf-Funktion mit batch-size etc.

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