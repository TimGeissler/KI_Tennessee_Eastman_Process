import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix

def matchPredictionsWithLabels(predictions, labels):
    # Diese Methode gibt ein Tupel aus, das einerseits aus den predictions und den zugehörigen Labeln besteht.
    # So wird sichergestellt, dass die ausgegebenen labels die gleiche Länge haben wie die predictions.

    predictions_vector = []
    labels_vector = []
    for i in range(len(predictions)):
        predictions_vector.append(np.argmax(predictions[i]))
        labels_vector.append(np.argmax(labels[i]))
    
    return np.array(predictions_vector), np.array(labels_vector)

def makeScatterPlot(predictions, labels, path):
    # In dieser Methode wird ein ScatterPlot aus den predictions und den labels erstellt.
    # Erst werden die labels geplottet, was eine 22-stufige Treppenstufe ergibt (grün gefärbt).
    # Im Anschluss werden die predictions des Modells darüber geplottet. So kann recht schnell sichtbar gemacht werden, wie viele fehlerhafte
    # Vorhersagen getroffen wurden. Je mehr Fehler gemacht wurden, desto mehr weicht der rote Plot von den grünen "Treppenstufen" ab.

    predictions_vector, y_test_vector = matchPredictionsWithLabels(predictions, labels)

    plt.scatter(range(len(predictions)),y_test_vector,c='g')
    plt.scatter(range(len(predictions)),predictions_vector,c='r')
    #plt.scatter(range(10560),y_test_vector,c='g')
    #plt.show()
    if not os.path.exists(path):
        plt.savefig(path)
    else:
        print('WARNING: file already exists and will not be overwritten. Please change the testNumber')

    plt.close()

def makeConfMatrixPlot(predictions, labels, path):
    # In dieser Methode wird eine ConfusionMatrix aus den predictions und den labels erstellt.
    # Durch die Farbgebung wird schnell sichtbar, wie viele fehlerhafte Vorhersagen getroffen wurden.
    # Je mehr Fehler gemacht wurden, desto mehr weicht der Plot von einer Winkelhalbierenden ab.
    # Darüber hinaus werden in diesem Plot Zahlen mit geplottet, sodass genau sichtbar ist, 
    # wie oft ein Teil eines Fehlerbildes für einen Teil eines bestimmten anderen gehalten wurde.

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

    plt.close()

def makeHistoryPlot(history, path):
    # In dieser Methode wird das History-Objekt ausgewertet, das von der Methode model.fit() zurückgegeben wird.
    # Um die Plots des Trainingsverlaufs möglichst gut lesbar zu machen und Wertebereiche nicht zu stark zu variieren, 
    # werden losses und accuracies in getrennten Plots aufgetragen.

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.legend()

    if not os.path.exists(path):
        plt.savefig(path + '_Losses.png')
    else:
        print('WARNING: file already exists and will not be overwritten. Please change the testNumber')

    plt.close()



    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    if not os.path.exists(path):
        plt.savefig(path + '_Accuracy.png')
    else:
        print('WARNING: file already exists and will not be overwritten. Please change the testNumber')

    plt.close()


