import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


import ReadData
import EditData
import TrainingData
import ModelDFF
import Evaluation

####################################################################################
# Organisatorisches
#RELPATH = 'C:\\Path\\if\\files\\are\\on\\your\\PC\\'
RELPATH = 'Dataset\\'

# Trainingsparameter

test_Number = 7
numberOfEpochs = 200
batch_Size = 1
learningRate = 0.001
numberOfHiddenLayers = 2
neuronsPerHiddenLayer = 128
dropout = 0.0

####################################################################################
### Funktion zum Einlesen der Dateien

data_raw, dataTE_raw = ReadData.loadDataFromDirectory(RELPATH, 480, 0, 480)


####################################################################################
### normieren
data_norm = EditData.datensatzNormieren(data_raw)
dataTE_norm = EditData.datensatzNormieren(dataTE_raw)

####################################################################################
### aus normierter Matrix Trainigsdaten erstellen

training_data = TrainingData.CreateDatasetWithLabels(data_norm)
test_data = TrainingData.CreateDatasetWithLabels(dataTE_norm)

####################################################################################
### Training and test data

x_train, y_train = TrainingData.GetXYFromDataset(training_data)
x_test, y_test = TrainingData.GetXYFromDataset(test_data)

####################################################################################
# Erstellen des Modells bzw. Neuronalen Netzes 

# Sequential Model -> Feed-Forward Model 
# model = tf.keras.models.Sequential()

# #model.add(tf.keras.layers.Flatten()) # noch keine Ahnung was das bedeutet (Nacharbeiten)
# model.add(tf.keras.layers.Dense(52, activation=tf.nn.relu))
# #model.add(tf.keras.layers.Dropout(0.2))#, input_shape = x_train.shape[1:]))
# # erste Schicht des NN mit 52 Neuronen und der Aktivierungsfunktion (z.B Sprungfunktion, Sigmoid-Funktion (Schwanenhals-Funktion))
# #model.add(tf.keras.layers.Dense(52, activation = tf.nn.relu)) # Vorlesung
# # zweite Schicht des NN mit 128 Neuronen und der Aktivierungsfunktion (z.B Sprungfunktion, Sigmoid-Funktion (Schwanenhals-Funktion))
# model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
# #model.add(tf.keras.layers.Dropout(0.2))
# model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
# #model.add(tf.keras.layers.Dropout(0.2))
# # Ausgangsschicht des NN mit 22 Neuronen, da es 21 Fälle gibt. ie Aktivierungsfunktion ist hier eine Wahrscheinlichkeitsverteilung
# model.add(tf.keras.layers.Dense(22, activation = tf.nn.softmax))

# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate),
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])

model = ModelDFF.ModelDFF(learningRate=learningRate, 
                          dynamicLearningRate=False, 
                          dropout=dropout, 
                          numberOfHiddenLayers=numberOfHiddenLayers, 
                          neuronsPerHiddenLayer=neuronsPerHiddenLayer).model

history = model.fit(x_train, 
          y_train, 
          epochs=numberOfEpochs,
          validation_data=(x_test, y_test),
          shuffle=True,
          batch_size=batch_Size
          )

path_hist_without_suffix = 'Plots_2\\History\\' + str(test_Number) 
Evaluation.makeHistoryPlot(history, path_hist_without_suffix)


predictions = model.predict([x_test])

path_scatter = 'Plots_2\\ScatterPlot\\' + str(test_Number) + '.png'
Evaluation.makeScatterPlot(predictions, y_test, path_scatter)

#Heatmap 
path_heatmap = 'Plots_2\\Heatmap (ConfusionMatrix)\\' + str(test_Number) + '.png'
Evaluation.makeConfMatrixPlot(predictions, y_test, path_heatmap)


print('done without error lol how come')