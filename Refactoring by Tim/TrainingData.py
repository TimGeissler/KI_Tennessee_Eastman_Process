import numpy as np

def CreateDatasetWithLabels(data):
    dataset = []

    for i in range(22):
    # hier label erzeugen -> one-hot vector
        oneHotVector = np.eye(22)[i, :]

        fault_data = np.transpose(data[i])

        for k in range(480):
            dataset.append([fault_data[k], oneHotVector])

    return dataset

def GetXYFromDataset(dataset):
    x = []
    y = []

    for testvalues, label in dataset:
        x.append(testvalues)
        y.append(label)

    return np.array(x), np.array(y)