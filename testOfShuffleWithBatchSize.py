import random
import numpy as np

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

testList = np.arange(25)
print(shuffleWithBatchSize(testList, 6))

print(round(5/2))