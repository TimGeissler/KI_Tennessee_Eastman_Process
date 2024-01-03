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

test_Number = 66
numberOfEpochs = 300

batch_Size = 20
batch_Size_Predict = batch_Size

learningRate = 0.001
dynamicLearningRate = False
decaySteps = 1000
decayRate = 0.98

numberOfHiddenLayers = 2
neuronsPerHiddenLayer = 256

dropout = 0.8

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
### Erstellen des Modells bzw. Neuronalen Netzes
model = ModelDFF.ModelDFF(learningRate=learningRate, 
                          dynamicLearningRate=False, 
                          decayRate=decayRate,
                          decaySteps=decaySteps,
                          dropout=dropout, 
                          numberOfHiddenLayers=numberOfHiddenLayers, 
                          neuronsPerHiddenLayer=neuronsPerHiddenLayer
                          ).model

history = model.fit(x_train, 
          y_train, 
          epochs=numberOfEpochs,
          validation_data=(x_test, y_test),
          shuffle=True,
          batch_size=batch_Size
          )

# History Plot
path_hist_without_suffix = 'Plots_2\\History\\' + str(test_Number) 
Evaluation.makeHistoryPlot(history, path_hist_without_suffix)


predictions = model.predict([x_test],
                            batch_size=batch_Size_Predict
                            )

# Scatterplot
path_scatter = 'Plots_2\\ScatterPlot\\' + str(test_Number) + '.png'
Evaluation.makeScatterPlot(predictions, y_test, path_scatter)

#Heatmap 
path_heatmap = 'Plots_2\\Heatmap (ConfusionMatrix)\\' + str(test_Number) + '.png'
Evaluation.makeConfMatrixPlot(predictions, y_test, path_heatmap)


print('done without error lol how come')