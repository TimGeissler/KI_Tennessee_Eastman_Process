import numpy as np

def CreateDatasetWithLabels(data):
    # Diese Funktion erstellt aus dem gegebenen Datensatz (22, 52, 480)
    # einen Trainingsdatensatz (10560, 52) 
    # und verknüpft diese Daten jeweils mit einem Label (One-Hot-Vektor).
    # Rückgabe erfolgt in Form eines Arrays.

    dataset = []

    for i in range(22):
    # hier label erzeugen -> one-hot vector
        oneHotVector = np.eye(22)[i, :]

        fault_data = np.transpose(data[i])

        for k in range(480):
            dataset.append([fault_data[k], oneHotVector])

    return dataset

def GetXYFromDataset(dataset):
    # Diese Funktion trennt den erstellten Datensatz wieder in Daten und Labels (x, y) auf, 
    # um ihn in korrekter Form dem Modell zuführen zu können.

    x = []
    y = []

    for testvalues, label in dataset:
        x.append(testvalues)
        y.append(label)

    return np.array(x), np.array(y)