import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix

def matchPredictionsWithLabels(predictions, labels):
    predictions_vector = []
    labels_vector = []
    for i in range(len(predictions)):
        predictions_vector.append(np.argmax(predictions[i]))
        labels_vector.append(np.argmax(labels[i]))
    
    return np.array(predictions_vector), np.array(labels_vector)

def makeScatterPlot(predictions, labels, path):
    predictions_vector, y_test_vector = matchPredictionsWithLabels(predictions, labels)

    plt.scatter(range(len(predictions)),y_test_vector,c='g')
    plt.scatter(range(len(predictions)),predictions_vector,c='r')
    #plt.scatter(range(10560),y_test_vector,c='g')
    #plt.show()
    if not os.path.exists(path):
        plt.savefig(path)
    else:
        print('WARNING: file already exists and will not be overwritten. Please change the testNumber')


def makeConfMatrixPlot(predictions, labels, path):
    predictions_vector, y_test_vector = matchPredictionsWithLabels(predictions, labels)

    conf_matrix = confusion_matrix(y_test_vector, predictions_vector)

    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
    plt.xlabel('Vorhergesagt')
    plt.ylabel('Tatsächlich')
    plt.title('Confusion Matrix')

    if not os.path.exists(path):
        plt.savefig(path)
    else:
        print('WARNING: file already exists and will not be overwritten. Please change the testNumber')



