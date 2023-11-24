#!/usr/bin/env python
# coding: utf-8

# In[34]:


import tensorflow as tf
import numpy as np
# TensorFlow wird nicht benötigt, es sei denn, Sie verwenden es anderswo in Ihrem Code
# import tensorflow as tf

# Öffnen der .dat-Datei im Lesemodus
file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d00.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()

        # Die Daten in eine Matrix umwandeln
        data_matrix = [list(map(float, line.split()[:480])) for line in lines]

    # Größe der neuen Matrix ausgeben
    new_matrix_size = (len(data_matrix), len(data_matrix[0]))
    print(f"Größe der Trainingsdaten Matrix: {new_matrix_size}")

except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')

    
#Einlesen der Testdaten 
file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d00_te.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()

        # Die Daten in eine Matrix umwandeln
        data_matrix_te0 = [list(map(float, line.split()[:52])) for line in lines]
        
        data_matrix_te0a = np.transpose(data_matrix_te0)
        
        data_matrix_te0a_reduced = data_matrix_te0a[:, :480]

        # Größe der neuen Matrix ausgeben
        new_matrix_size_te0a_reduced = (len(data_matrix_te0a_reduced), len(data_matrix_te0a_reduced[0]))
        print(f"Größe der Testdaten Matrix: {new_matrix_size_te0a_reduced}")

except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


# In[35]:


file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d01.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()

        # Die Daten in eine Matrix umwandeln
        data_matrix1a = [list(map(float, line.split())) for line in lines]
    
        data_matrix1 = np.transpose(data_matrix1a)

        # Größe der Matrix ausgeben
        matrix_size = (len(data_matrix1), len(data_matrix1[0]))
        print(f"Größe der Matrix: {matrix_size}")

except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


#Einlesen der Testdaten 
file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d01_te.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()
        
         # Die Daten in eine Matrix umwandeln
        data_matrix_te1 = [list(map(float, line.split()[:52])) for line in lines]
        
        data_matrix_te1a = np.transpose(data_matrix_te1)
        
        data_matrix_te1a_reduced = data_matrix_te1a[:, :480]
        
        # Größe der neuen Matrix ausgeben
        new_matrix_size_te1a_reduced = (len(data_matrix_te1a_reduced), len(data_matrix_te1a_reduced[0]))
        print(f"Größe der Testdaten Matrix: {new_matrix_size_te1a_reduced}")
    
except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


# In[36]:


file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d02.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()

        # Die Daten in eine Matrix umwandeln
        data_matrix2a = [list(map(float, line.split())) for line in lines]
    
    data_matrix2 = np.transpose(data_matrix2a)

    # Größe der Matrix ausgeben
    matrix_size = (len(data_matrix2), len(data_matrix2[0]))
    print(f"Größe der Matrix: {matrix_size}")

except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')




    #Einlesen der Testdaten 
file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d02_te.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()
        
         # Die Daten in eine Matrix umwandeln
        data_matrix_te2 = [list(map(float, line.split()[:52])) for line in lines]
        
        data_matrix_te2a = np.transpose(data_matrix_te2)
        
        data_matrix_te2a_reduced = data_matrix_te2a[:, :480]
        
        # Größe der neuen Matrix ausgeben
        new_matrix_size_te2a_reduced = (len(data_matrix_te2a_reduced), len(data_matrix_te2a_reduced[0]))
        print(f"Größe der Testdaten Matrix: {new_matrix_size_te2a_reduced}")
    
except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


# In[37]:


file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d03.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()

        # Die Daten in eine Matrix umwandeln
        data_matrix3a = [list(map(float, line.split())) for line in lines]
    
    data_matrix3 = np.transpose(data_matrix3a)

    # Größe der Matrix ausgeben
    matrix_size = (len(data_matrix3), len(data_matrix3[0]))
    print(f"Größe der Matrix: {matrix_size}")

except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


    
    
    #Einlesen der Testdaten 
file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d03_te.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()
        
         # Die Daten in eine Matrix umwandeln
        data_matrix_te3 = [list(map(float, line.split()[:52])) for line in lines]
        
        data_matrix_te3a = np.transpose(data_matrix_te3)
        
        data_matrix_te3a_reduced = data_matrix_te3a[:, :480]
        
        # Größe der neuen Matrix ausgeben
        new_matrix_size_te3a_reduced = (len(data_matrix_te3a_reduced), len(data_matrix_te3a_reduced[0]))
        print(f"Größe der Testdaten Matrix: {new_matrix_size_te3a_reduced}")
    
except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


# In[38]:


file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d04.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()

        # Die Daten in eine Matrix umwandeln
        data_matrix4a = [list(map(float, line.split())) for line in lines]
    
    data_matrix4 = np.transpose(data_matrix3a)

    # Größe der Matrix ausgeben
    matrix_size = (len(data_matrix4), len(data_matrix4[0]))
    print(f"Größe der Matrix: {matrix_size}")

except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')



    
#Einlesen der Testdaten 
file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d04_te.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()
        
         # Die Daten in eine Matrix umwandeln
        data_matrix_te4 = [list(map(float, line.split()[:52])) for line in lines]
        
        data_matrix_te4a = np.transpose(data_matrix_te4)
        
        data_matrix_te4a_reduced = data_matrix_te4a[:, :480]
        
        # Größe der neuen Matrix ausgeben
        new_matrix_size_te4a_reduced = (len(data_matrix_te4a_reduced), len(data_matrix_te4a_reduced[0]))
        print(f"Größe der Testdaten Matrix: {new_matrix_size_te4a_reduced}")
    
except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


# In[39]:


file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d05.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()

        # Die Daten in eine Matrix umwandeln
        data_matrix5a = [list(map(float, line.split())) for line in lines]
    
    data_matrix5 = np.transpose(data_matrix5a)

    # Größe der Matrix ausgeben
    matrix_size = (len(data_matrix5), len(data_matrix5[0]))
    print(f"Größe der Matrix: {matrix_size}")

except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


    
    
    
#Einlesen der Testdaten 
file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d05_te.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()
        
         # Die Daten in eine Matrix umwandeln
        data_matrix_te5 = [list(map(float, line.split()[:52])) for line in lines]
        
        data_matrix_te5a = np.transpose(data_matrix_te5)
        
        data_matrix_te5a_reduced = data_matrix_te5a[:, :480]
        
        # Größe der neuen Matrix ausgeben
        new_matrix_size_te5a_reduced = (len(data_matrix_te5a_reduced), len(data_matrix_te5a_reduced[0]))
        print(f"Größe der Testdaten Matrix: {new_matrix_size_te5a_reduced}")
    
except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


# In[40]:


file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d06.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()

        # Die Daten in eine Matrix umwandeln
        data_matrix6a = [list(map(float, line.split())) for line in lines]
    
    data_matrix6 = np.transpose(data_matrix3a)

    # Größe der Matrix ausgeben
    matrix_size = (len(data_matrix6), len(data_matrix6[0]))
    print(f"Größe der Matrix: {matrix_size}")

except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')



    
#Einlesen der Testdaten 
file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d06_te.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()
        
         # Die Daten in eine Matrix umwandeln
        data_matrix_te6 = [list(map(float, line.split()[:52])) for line in lines]
        
        data_matrix_te6a = np.transpose(data_matrix_te6)
        
        data_matrix_te6a_reduced = data_matrix_te6a[:, :480]
        
        # Größe der neuen Matrix ausgeben
        new_matrix_size_te6a_reduced = (len(data_matrix_te6a_reduced), len(data_matrix_te6a_reduced[0]))
        print(f"Größe der Testdaten Matrix: {new_matrix_size_te6a_reduced}")
    
except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


# In[41]:


file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d07.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()

        # Die Daten in eine Matrix umwandeln
        data_matrix7a = [list(map(float, line.split())) for line in lines]
    
    data_matrix7 = np.transpose(data_matrix7a)

    # Größe der Matrix ausgeben
    matrix_size = (len(data_matrix7), len(data_matrix7[0]))
    print(f"Größe der Matrix: {matrix_size}")

except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


    
    
#Einlesen der Testdaten 
file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d07_te.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()
        
         # Die Daten in eine Matrix umwandeln
        data_matrix_te7 = [list(map(float, line.split()[:52])) for line in lines]
        
        data_matrix_te7a = np.transpose(data_matrix_te7)
        
        data_matrix_te7a_reduced = data_matrix_te7a[:, :480]
        
        # Größe der neuen Matrix ausgeben
        new_matrix_size_te7a_reduced = (len(data_matrix_te7a_reduced), len(data_matrix_te7a_reduced[0]))
        print(f"Größe der Testdaten Matrix: {new_matrix_size_te7a_reduced}")
    
except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


# In[25]:


file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d08.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()

        # Die Daten in eine Matrix umwandeln
        data_matrix8a = [list(map(float, line.split())) for line in lines]
    
    data_matrix8 = np.transpose(data_matrix8a)

    # Größe der Matrix ausgeben
    matrix_size = (len(data_matrix8), len(data_matrix8[0]))
    print(f"Größe der Matrix: {matrix_size}")

except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


    
    
#Einlesen der Testdaten 
file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d08_te.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()
        
         # Die Daten in eine Matrix umwandeln
        data_matrix_te8 = [list(map(float, line.split()[:52])) for line in lines]
        
        data_matrix_te8a = np.transpose(data_matrix_te8)
        
        data_matrix_te8a_reduced = data_matrix_te8a[:, :480]
        
        # Größe der neuen Matrix ausgeben
        new_matrix_size_te8a_reduced = (len(data_matrix_te8a_reduced), len(data_matrix_te8a_reduced[0]))
        print(f"Größe der Testdaten Matrix: {new_matrix_size_te8a_reduced}")
    
except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


# In[26]:


file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d09.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()

        # Die Daten in eine Matrix umwandeln
        data_matrix9a = [list(map(float, line.split())) for line in lines]
    
    data_matrix9 = np.transpose(data_matrix9a)

    # Größe der Matrix ausgeben
    matrix_size = (len(data_matrix9), len(data_matrix9[0]))
    print(f"Größe der Matrix: {matrix_size}")

except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


    
    
#Einlesen der Testdaten 
file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d09_te.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()
        
         # Die Daten in eine Matrix umwandeln
        data_matrix_te9 = [list(map(float, line.split()[:52])) for line in lines]
        
        data_matrix_te9a = np.transpose(data_matrix_te9)
        
        data_matrix_te9a_reduced = data_matrix_te9a[:, :480]
        
        # Größe der neuen Matrix ausgeben
        new_matrix_size_te9a_reduced = (len(data_matrix_te9a_reduced), len(data_matrix_te9a_reduced[0]))
        print(f"Größe der Testdaten Matrix: {new_matrix_size_te9a_reduced}")
    
except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


# In[27]:


file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d10.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()

        # Die Daten in eine Matrix umwandeln
        data_matrix10a = [list(map(float, line.split())) for line in lines]
    
    data_matrix10 = np.transpose(data_matrix10a)

    # Größe der Matrix ausgeben
    matrix_size = (len(data_matrix10), len(data_matrix10[0]))
    print(f"Größe der Matrix: {matrix_size}")

except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


    
    
#Einlesen der Testdaten 
file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d10_te.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()
        
         # Die Daten in eine Matrix umwandeln
        data_matrix_te10 = [list(map(float, line.split()[:52])) for line in lines]
        
        data_matrix_te10a = np.transpose(data_matrix_te10)
        
        data_matrix_te10a_reduced = data_matrix_te10a[:, :480]
        
        # Größe der neuen Matrix ausgeben
        new_matrix_size_te10a_reduced = (len(data_matrix_te10a_reduced), len(data_matrix_te10a_reduced[0]))
        print(f"Größe der Testdaten Matrix: {new_matrix_size_te10a_reduced}")
    
except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


# In[28]:


file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d11.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()

        # Die Daten in eine Matrix umwandeln
        data_matrix11a = [list(map(float, line.split())) for line in lines]
    
    data_matrix11 = np.transpose(data_matrix11a)

    # Größe der Matrix ausgeben
    matrix_size = (len(data_matrix11), len(data_matrix11[0]))
    print(f"Größe der Matrix: {matrix_size}")

except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


    
    
#Einlesen der Testdaten 
file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d11_te.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()
        
         # Die Daten in eine Matrix umwandeln
        data_matrix_te11 = [list(map(float, line.split()[:52])) for line in lines]
        
        data_matrix_te11a = np.transpose(data_matrix_te11)
        
        data_matrix_te11a_reduced = data_matrix_te11a[:, :480]
        
        # Größe der neuen Matrix ausgeben
        new_matrix_size_te11a_reduced = (len(data_matrix_te11a_reduced), len(data_matrix_te11a_reduced[0]))
        print(f"Größe der Testdaten Matrix: {new_matrix_size_te11a_reduced}")
    
except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


# In[29]:


file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d12.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()

        # Die Daten in eine Matrix umwandeln
        data_matrix12a = [list(map(float, line.split())) for line in lines]
    
    data_matrix12 = np.transpose(data_matrix12a)

    # Größe der Matrix ausgeben
    matrix_size = (len(data_matrix12), len(data_matrix12[0]))
    print(f"Größe der Matrix: {matrix_size}")

except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


    
    
#Einlesen der Testdaten 
file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d12_te.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()
        
         # Die Daten in eine Matrix umwandeln
        data_matrix_te12 = [list(map(float, line.split()[:52])) for line in lines]
        
        data_matrix_te12a = np.transpose(data_matrix_te12)
        
        data_matrix_te12a_reduced = data_matrix_te12a[:, :480]
        
        # Größe der neuen Matrix ausgeben
        new_matrix_size_te12a_reduced = (len(data_matrix_te12a_reduced), len(data_matrix_te12a_reduced[0]))
        print(f"Größe der Testdaten Matrix: {new_matrix_size_te12a_reduced}")
    
except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


# In[30]:


file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d13.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()

        # Die Daten in eine Matrix umwandeln
        data_matrix13a = [list(map(float, line.split())) for line in lines]
    
    data_matrix13 = np.transpose(data_matrix13a)

    # Größe der Matrix ausgeben
    matrix_size = (len(data_matrix13), len(data_matrix13[0]))
    print(f"Größe der Matrix: {matrix_size}")

except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


    
    
#Einlesen der Testdaten 
file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d13_te.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()
        
         # Die Daten in eine Matrix umwandeln
        data_matrix_te13 = [list(map(float, line.split()[:52])) for line in lines]
        
        data_matrix_te13a = np.transpose(data_matrix_te13)
        
        data_matrix_te13a_reduced = data_matrix_te13a[:, :480]
        
        # Größe der neuen Matrix ausgeben
        new_matrix_size_te13a_reduced = (len(data_matrix_te13a_reduced), len(data_matrix_te13a_reduced[0]))
        print(f"Größe der Testdaten Matrix: {new_matrix_size_te13a_reduced}")
    
except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


# In[31]:


file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d14.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()

        # Die Daten in eine Matrix umwandeln
        data_matrix14a = [list(map(float, line.split())) for line in lines]
    
    data_matrix14 = np.transpose(data_matrix14a)

    # Größe der Matrix ausgeben
    matrix_size = (len(data_matrix14), len(data_matrix14[0]))
    print(f"Größe der Matrix: {matrix_size}")

except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


    
    
#Einlesen der Testdaten 
file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d14_te.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()
        
         # Die Daten in eine Matrix umwandeln
        data_matrix_te14 = [list(map(float, line.split()[:52])) for line in lines]
        
        data_matrix_te14a = np.transpose(data_matrix_te14)
        
        data_matrix_te14a_reduced = data_matrix_te14a[:, :480]
        
        # Größe der neuen Matrix ausgeben
        new_matrix_size_te14a_reduced = (len(data_matrix_te14a_reduced), len(data_matrix_te14a_reduced[0]))
        print(f"Größe der Testdaten Matrix: {new_matrix_size_te14a_reduced}")
    
except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


# In[32]:


file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d15.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()

        # Die Daten in eine Matrix umwandeln
        data_matrix15a = [list(map(float, line.split())) for line in lines]
    
    data_matrix15 = np.transpose(data_matrix15a)

    # Größe der Matrix ausgeben
    matrix_size = (len(data_matrix15), len(data_matrix15[0]))
    print(f"Größe der Matrix: {matrix_size}")

except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


    
#Einlesen der Testdaten 
file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d15_te.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()
        
         # Die Daten in eine Matrix umwandeln
        data_matrix_te15 = [list(map(float, line.split()[:52])) for line in lines]
        
        data_matrix_te15a = np.transpose(data_matrix_te15)
        
        data_matrix_te15a_reduced = data_matrix_te15a[:, :480]
        
        # Größe der neuen Matrix ausgeben
        new_matrix_size_te15a_reduced = (len(data_matrix_te15a_reduced), len(data_matrix_te15a_reduced[0]))
        print(f"Größe der Testdaten Matrix: {new_matrix_size_te15a_reduced}")
    
except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


# In[33]:


file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d16.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()

        # Die Daten in eine Matrix umwandeln
        data_matrix16a = [list(map(float, line.split())) for line in lines]
    
    data_matrix16 = np.transpose(data_matrix16a)

    # Größe der Matrix ausgeben
    matrix_size = (len(data_matrix16), len(data_matrix16[0]))
    print(f"Größe der Matrix: {matrix_size}")

except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


    
    
#Einlesen der Testdaten 
file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d16_te.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()
        
         # Die Daten in eine Matrix umwandeln
        data_matrix_te16 = [list(map(float, line.split()[:52])) for line in lines]
        
        data_matrix_te16a = np.transpose(data_matrix_te16)
        
        data_matrix_te16a_reduced = data_matrix_te16a[:, :480]
        
        # Größe der neuen Matrix ausgeben
        new_matrix_size_te16a_reduced = (len(data_matrix_te16a_reduced), len(data_matrix_te16a_reduced[0]))
        print(f"Größe der Testdaten Matrix: {new_matrix_size_te16a_reduced}")
    
except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


# In[34]:


file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d17.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()

        # Die Daten in eine Matrix umwandeln
        data_matrix17a = [list(map(float, line.split())) for line in lines]
    
    data_matrix17 = np.transpose(data_matrix17a)

    # Größe der Matrix ausgeben
    matrix_size = (len(data_matrix17), len(data_matrix17[0]))
    print(f"Größe der Matrix: {matrix_size}")

except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


    
    
#Einlesen der Testdaten 
file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d17_te.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()
        
         # Die Daten in eine Matrix umwandeln
        data_matrix_te17 = [list(map(float, line.split()[:52])) for line in lines]
        
        data_matrix_te17a = np.transpose(data_matrix_te17)
        
        data_matrix_te17a_reduced = data_matrix_te17a[:, :480]
        
        # Größe der neuen Matrix ausgeben
        new_matrix_size_te17a_reduced = (len(data_matrix_te17a_reduced), len(data_matrix_te17a_reduced[0]))
        print(f"Größe der Testdaten Matrix: {new_matrix_size_te17a_reduced}")
    
except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


# In[35]:


file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d18.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()

        # Die Daten in eine Matrix umwandeln
        data_matrix18a = [list(map(float, line.split())) for line in lines]
    
    data_matrix18 = np.transpose(data_matrix18a)

    # Größe der Matrix ausgeben
    matrix_size = (len(data_matrix18), len(data_matrix18[0]))
    print(f"Größe der Matrix: {matrix_size}")

except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


    
    
#Einlesen der Testdaten 
file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d18_te.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()
        
         # Die Daten in eine Matrix umwandeln
        data_matrix_te18 = [list(map(float, line.split()[:52])) for line in lines]
        
        data_matrix_te18a = np.transpose(data_matrix_te18)
        
        data_matrix_te18a_reduced = data_matrix_te18a[:, :480]
        
        # Größe der neuen Matrix ausgeben
        new_matrix_size_te18a_reduced = (len(data_matrix_te18a_reduced), len(data_matrix_te18a_reduced[0]))
        print(f"Größe der Testdaten Matrix: {new_matrix_size_te18a_reduced}")
    
except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


# In[36]:


file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d19.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()

        # Die Daten in eine Matrix umwandeln
        data_matrix19a = [list(map(float, line.split())) for line in lines]
    
    data_matrix19 = np.transpose(data_matrix19a)

    # Größe der Matrix ausgeben
    matrix_size = (len(data_matrix19), len(data_matrix19[0]))
    print(f"Größe der Matrix: {matrix_size}")

except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


    
    
#Einlesen der Testdaten 
file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d19_te.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()
        
         # Die Daten in eine Matrix umwandeln
        data_matrix_te19 = [list(map(float, line.split()[:52])) for line in lines]
        
        data_matrix_te19a = np.transpose(data_matrix_te19)
        
        data_matrix_te19a_reduced = data_matrix_te19a[:, :480]
        
        # Größe der neuen Matrix ausgeben
        new_matrix_size_te19a_reduced = (len(data_matrix_te19a_reduced), len(data_matrix_te19a_reduced[0]))
        print(f"Größe der Testdaten Matrix: {new_matrix_size_te19a_reduced}")
    
except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


# In[37]:


file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d20.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()

        # Die Daten in eine Matrix umwandeln
        data_matrix20a = [list(map(float, line.split())) for line in lines]
    
    data_matrix20 = np.transpose(data_matrix20a)

    # Größe der Matrix ausgeben
    matrix_size = (len(data_matrix20), len(data_matrix20[0]))
    print(f"Größe der Matrix: {matrix_size}")

except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


    
    
#Einlesen der Testdaten 
file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d20_te.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()
        
         # Die Daten in eine Matrix umwandeln
        data_matrix_te20 = [list(map(float, line.split()[:52])) for line in lines]
        
        data_matrix_te20a = np.transpose(data_matrix_te20)
        
        data_matrix_te20a_reduced = data_matrix_te20a[:, :480]
        
        # Größe der neuen Matrix ausgeben
        new_matrix_size_te20a_reduced = (len(data_matrix_te20a_reduced), len(data_matrix_te20a_reduced[0]))
        print(f"Größe der Testdaten Matrix: {new_matrix_size_te20a_reduced}")
    
except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


# In[38]:


file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d21.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()

        # Die Daten in eine Matrix umwandeln
        data_matrix21a = [list(map(float, line.split())) for line in lines]
    
    data_matrix21 = np.transpose(data_matrix21a)

    # Größe der Matrix ausgeben
    matrix_size = (len(data_matrix21), len(data_matrix21[0]))
    print(f"Größe der Matrix: {matrix_size}")

except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


    
    
#Einlesen der Testdaten 
file_path = '/Users/justinolivieri/Desktop/Künstliche Intelligenz/d21_te.dat'

try:
    with open(file_path, 'r') as file:
        # Zeilen aus der Datei lesen und in eine Liste von Zeichenketten aufteilen
        lines = file.read().splitlines()
        
         # Die Daten in eine Matrix umwandeln
        data_matrix_te21 = [list(map(float, line.split()[:52])) for line in lines]
        
        data_matrix_te21a = np.transpose(data_matrix_te21)
        
        data_matrix_te21a_reduced = data_matrix_te21a[:, :480]
        
        # Größe der neuen Matrix ausgeben
        new_matrix_size_te21a_reduced = (len(data_matrix_te21a_reduced), len(data_matrix_te21a_reduced[0]))
        print(f"Größe der Testdaten Matrix: {new_matrix_size_te21a_reduced}")
    
except FileNotFoundError:
    print(f'Die Datei {file_path} wurde nicht gefunden.')

except Exception as e:
    print(f'Fehler beim Lesen der Datei: {str(e)}')


# In[17]:


x_train = data_matrix
x_train = np.array(x_train)
y_train = np.random.randint(21, size=(x_train.shape[0],))

x_test = data_matrix_te0a_reduced 


# In[29]:


# Erstellen des Modells bzw. Neuronalen Netzes 

# Sequential Model -> Feed-Forward Model 
model = tf.keras.models.Sequential()


model.add(tf.keras.layers.Flatten()) # noch keine Ahnung was das bedeutet (Nacharbeiten)

# erste Schicht des NN mit 128 Neuronen und der Aktivierungsfunktion (z.B Sprungfunktion, Sigmoid-Funktion (Schwanenhals-Funktion))
model.add(tf.keras.layers.Dense(52, activation=tf.nn.relu)) # Vorlesung
# zweite Schicht des NN mit 128 Neuronen und der Aktivierungsfunktion (z.B Sprungfunktion, Sigmoid-Funktion (Schwanenhals-Funktion))
model.add(tf.keras.layers.Dense(52, activation=tf.nn.relu))
# Ausgangsschicht des NN mit 21 Neuronen da es 21 Fälle gibt, die Aktivierungsfunktion ist diesmal eine Wahrscheinlichkeitsverteilung
model.add(tf.keras.layers.Dense(21, activation=tf.nn.softmax)) #stimmt so noch nicht mit 21 Fällen 

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])


# In[33]:


model.fit(x_train, y_train, epochs=3) # Hier scheint es ein problem damit zu geben das lable0 


# In[ ]:




