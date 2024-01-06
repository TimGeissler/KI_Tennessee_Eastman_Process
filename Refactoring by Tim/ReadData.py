import numpy as np
import os

def loadDataFromFile(path, input_end, crop_start, crop_stop):
    # Funktion zum Einlesen eines Datensatzes, der sich in einer Datei befindet und Rückgabe als Matrix.

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
        return data

    except FileNotFoundError:
        print(f'Die Datei {path} wurde nicht gefunden.')

    except Exception as e:
        print(f'Fehler beim Lesen der Datei: {str(e)}')

def loadDataFromDirectory(path, input_end, crop_start, crop_stop):
    # Einlesen von allen Datensätzen aus allen Dateien in einem Verzeichnis.
    # Dazu werden sie je nach Dateiname in Trainings- und Testdaten aufgeteilt und in Form von zwei Arrays zurückgegeben.
    
    data = []
    dataTE = []

    # Dateien einlesen
    for datei in sorted(os.listdir(path)):
        filePath = path + datei

        if 'te' in datei:
            data_matrix = loadDataFromFile(filePath, input_end, crop_start, crop_stop)
            dataTE.append(data_matrix)
        else:
            data_matrix = loadDataFromFile(filePath, input_end, crop_start, crop_stop)
            data.append(data_matrix)

    return np.array(data), np.array(dataTE)


