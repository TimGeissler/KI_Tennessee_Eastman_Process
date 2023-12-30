import numpy as np
import random



def datensatzNormieren(array):
    dataInput = np.array(array)
    data_norm = []
    
    for i in range(len(dataInput)):
        array2ndD = []

        for k in range(len(dataInput[i])):
            normOfSensorData = np.linalg.norm(dataInput[i][k])
            meanOfSensorData = np.mean(dataInput[i][k])
        
            array3rdD = []
            
            for m in range(len(dataInput[i][k])):
                array3rdD.append((dataInput[i][k][m] - meanOfSensorData)/normOfSensorData)

            array2ndD.append(array3rdD)

        data_norm.append(array2ndD)

    return np.array(data_norm)

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

