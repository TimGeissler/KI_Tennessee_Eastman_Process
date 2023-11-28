#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[4]:


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


# In[3]:


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


# In[4]:


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


# In[5]:


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


# In[6]:


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


# In[7]:


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


# In[8]:


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


# In[9]:


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


# In[10]:


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


# In[11]:


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


# In[12]:


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


# In[13]:


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


# In[14]:


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


# In[15]:


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


# In[16]:


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


# In[17]:


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


# In[18]:


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


# In[19]:


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


# In[20]:


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


# In[21]:


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


# In[22]:


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


# In[7]:


x_train = data_matrix
x_train = np.array(x_train)
x_train_normalized = np.array(x_train) / 255.0

y_train = np.zeros(x_train.shape[0], dtype=int)

x_test = data_matrix_te0a_reduced 


# In[5]:


# Erstellen des Modells bzw. Neuronalen Netzes 

# Sequential Model -> Feed-Forward Model 
model = tf.keras.models.Sequential()


model.add(tf.keras.layers.Flatten()) # noch keine Ahnung was das bedeutet (Nacharbeiten)

# erste Schicht des NN mit 128 Neuronen und der Aktivierungsfunktion (z.B Sprungfunktion, Sigmoid-Funktion (Schwanenhals-Funktion))
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu)) # Vorlesung
# zweite Schicht des NN mit 128 Neuronen und der Aktivierungsfunktion (z.B Sprungfunktion, Sigmoid-Funktion (Schwanenhals-Funktion))
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
# dritte Schicht des NN mit 128 Neuronen und der Aktivierungsfunktion (z.B Sprungfunktion, Sigmoid-Funktion (Schwanenhals-Funktion))
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
# Ausgangsschicht des NN mit 21 Neuronen da es 21 Fälle gibt, die Aktivierungsfunktion ist diesmal eine Wahrscheinlichkeitsverteilung
model.add(tf.keras.layers.Dense(21, activation=tf.nn.softmax)) #stimmt so noch nicht mit 21 Fällen 

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])


# In[8]:


model.fit(x_train_normalized, y_train, epochs=10) # Hier scheint es ein problem damit zu geben das lable0 


# In[26]:


x_train1 = data_matrix1
x_train1 = np.array(x_train1)
x_train_normalized1 = np.array(x_train1) / 255.0

y_train1 = np.ones(x_train1.shape[0], dtype=int)

x_test1 = data_matrix_te1a_reduced 


# In[27]:


model.fit(x_train_normalized1, y_train1, epochs=10) # Hier scheint es ein problem damit zu geben das lable1 


# In[28]:


x_train2 = data_matrix2
x_train2 = np.array(x_train2)
x_train_normalized2 = np.array(x_train2) / 255.0

y_train2 = np.full(x_train2.shape[0], 2, dtype=int)

x_test2 = data_matrix_te2a_reduced 


# In[29]:


model.fit(x_train_normalized2, y_train2, epochs=10) # Hier scheint es ein problem damit zu geben das lable2 


# In[30]:


x_train3 = data_matrix3
x_train3 = np.array(x_train3)
x_train_normalized3 = np.array(x_train3) / 255.0

y_train3 = np.full(x_train3.shape[0], 3, dtype=int)

x_test2 = data_matrix_te3a_reduced 


# In[31]:


model.fit(x_train_normalized3, y_train3, epochs=10) # Hier scheint es ein problem damit zu geben das lable3 


# In[32]:


x_train4 = data_matrix4
x_train4 = np.array(x_train4)
x_train_normalized4 = np.array(x_train4) / 255.0

y_train4 = np.full(x_train4.shape[0], 4, dtype=int)

x_test2 = data_matrix_te4a_reduced 


# In[33]:


model.fit(x_train_normalized4, y_train4, epochs=10) # Hier scheint es ein problem damit zu geben das lable3 


# In[34]:


x_train5 = data_matrix5
x_train5 = np.array(x_train5)
x_train_normalized5 = np.array(x_train5) / 255.0

y_train5 = np.full(x_train5.shape[0], 5, dtype=int)

x_test5 = data_matrix_te5a_reduced 


# In[35]:


model.fit(x_train_normalized5, y_train5, epochs=10) # Hier scheint es ein problem damit zu geben das lable3 


# In[36]:


x_train6 = data_matrix6
x_train6 = np.array(x_train6)
x_train_normalized6 = np.array(x_train6) / 255.0

y_train6 = np.full(x_train6.shape[0], 6, dtype=int)

x_test6 = data_matrix_te6a_reduced 


# In[37]:


model.fit(x_train_normalized6, y_train6, epochs=10) # Hier scheint es ein problem damit zu geben das lable3 


# In[38]:


x_train7 = data_matrix7
x_train7 = np.array(x_train7)
x_train_normalized7 = np.array(x_train7) / 255.0

y_train7 = np.full(x_train7.shape[0], 7, dtype=int)

x_test7 = data_matrix_te7a_reduced 


# In[39]:


model.fit(x_train_normalized7, y_train7, epochs=10) # Hier scheint es ein problem damit zu geben das lable3 


# In[40]:


x_train8 = data_matrix8
x_train8 = np.array(x_train8)
x_train_normalized8 = np.array(x_train8) / 255.0

y_train8 = np.full(x_train8.shape[0], 8, dtype=int)

x_test8 = data_matrix_te8a_reduced 


# In[41]:


model.fit(x_train_normalized8, y_train8, epochs=40) # Hier scheint es ein problem damit zu geben das lable3 


# In[42]:


x_train9 = data_matrix9
x_train9 = np.array(x_train9)
x_train_normalized9 = np.array(x_train9) / 255.0

y_train9 = np.full(x_train9.shape[0], 9, dtype=int)

x_test8 = data_matrix_te9a_reduced 


# In[43]:


model.fit(x_train_normalized9, y_train9, epochs=50) # Hier scheint es ein problem damit zu geben das lable3 


# In[44]:


x_train10 = data_matrix10
x_train10 = np.array(x_train10)
x_train_normalized10 = np.array(x_train10) / 255.0

y_train10 = np.full(x_train10.shape[0], 10, dtype=int)

x_test10 = data_matrix_te10a_reduced 


# In[45]:


model.fit(x_train_normalized10, y_train10, epochs=60) # Hier scheint es ein problem damit zu geben das lable3 


# In[46]:


x_train11 = data_matrix11
x_train11 = np.array(x_train11)
x_train_normalized11 = np.array(x_train11) / 255.0

y_train11 = np.full(x_train11.shape[0], 11, dtype=int)

x_test11 = data_matrix_te11a_reduced 


# In[47]:


model.fit(x_train_normalized11, y_train11, epochs=70) # Hier scheint es ein problem damit zu geben das lable11


# In[48]:


x_train12 = data_matrix12
x_train12 = np.array(x_train12)
x_train_normalized12 = np.array(x_train12) / 255.0

y_train12 = np.full(x_train12.shape[0], 12, dtype=int)

x_test12 = data_matrix_te12a_reduced 


# In[49]:


model.fit(x_train_normalized12, y_train12, epochs=80) # Hier scheint es ein problem damit zu geben das lable11


# In[50]:


x_train13 = data_matrix13
x_train13 = np.array(x_train13)
x_train_normalized13 = np.array(x_train13) / 255.0

y_train13 = np.full(x_train13.shape[0], 13, dtype=int)

x_test13 = data_matrix_te13a_reduced 


# In[51]:


model.fit(x_train_normalized13, y_train13, epochs=70) # Hier scheint es ein problem damit zu geben das lable11


# In[52]:


x_train14 = data_matrix14
x_train14 = np.array(x_train14)
x_train_normalized14 = np.array(x_train14) / 255.0

y_train14 = np.full(x_train14.shape[0], 14, dtype=int)

x_test14 = data_matrix_te14a_reduced 


# In[53]:


model.fit(x_train_normalized14, y_train14, epochs=60) # Hier scheint es ein problem damit zu geben das lable11


# In[54]:


x_train15 = data_matrix15
x_train15 = np.array(x_train15)
x_train_normalized15 = np.array(x_train15) / 255.0

y_train15 = np.full(x_train15.shape[0], 15, dtype=int)

x_test15 = data_matrix_te15a_reduced 


# In[55]:


model.fit(x_train_normalized15, y_train15, epochs=55) # Hier scheint es ein problem damit zu geben das lable11


# In[56]:


x_train16 = data_matrix16
x_train16 = np.array(x_train16)
x_train_normalized16 = np.array(x_train16) / 255.0

y_train16 = np.full(x_train15.shape[0], 16, dtype=int)

x_test16 = data_matrix_te16a_reduced 


# In[57]:


model.fit(x_train_normalized16, y_train16, epochs=55) # Hier scheint es ein problem damit zu geben das lable11


# In[58]:


x_train17 = data_matrix17
x_train17 = np.array(x_train17)
x_train_normalized17 = np.array(x_train17) / 255.0

y_train17 = np.full(x_train17.shape[0], 17, dtype=int)

x_test17 = data_matrix_te17a_reduced 


# In[59]:


model.fit(x_train_normalized17, y_train17, epochs=55) # Hier scheint es ein problem damit zu geben das lable11


# In[60]:


x_train18 = data_matrix18
x_train18 = np.array(x_train18)
x_train_normalized18 = np.array(x_train18) / 255.0

y_train18 = np.full(x_train18.shape[0], 18, dtype=int)

x_test18 = data_matrix_te18a_reduced 


# In[61]:


model.fit(x_train_normalized18, y_train18, epochs=50) # Hier scheint es ein problem damit zu geben das lable11


# In[62]:


x_train19 = data_matrix19
x_train19 = np.array(x_train19)
x_train_normalized19 = np.array(x_train19) / 255.0

y_train19 = np.full(x_train19.shape[0], 19, dtype=int)

x_test18 = data_matrix_te19a_reduced 


# In[63]:


model.fit(x_train_normalized19, y_train19, epochs=50) # Hier scheint es ein problem damit zu geben das lable11


# In[64]:


x_train20 = data_matrix20
x_train20 = np.array(x_train20)
x_train_normalized20 = np.array(x_train20) / 255.0

y_train20 = np.full(x_train20.shape[0], 20, dtype=int)

x_test20 = data_matrix_te20a_reduced 


# In[65]:


model.fit(x_train_normalized20, y_train20, epochs=50) # Hier scheint es ein problem damit zu geben das lable11


# In[66]:


x_train21 = data_matrix21
x_train21 = np.array(x_train21)
x_train_normalized21 = np.array(x_train21) / 255.0

y_train21 = np.full(x_train21.shape[0], 21, dtype=int)

x_test21 = data_matrix_te21a_reduced 


# In[67]:


model.fit(x_train_normalized21, y_train21, epochs=50) # Hier scheint es ein problem damit zu geben das lable11


# In[9]:


model.save('DFF_NN.model')


# In[10]:


new_model = tf.keras.models.load_model('DFF_NN')


# In[11]:


predictions = new_model.predict([x_test])


# In[12]:


print(predictions)


# In[13]:


print(np.argmax(predictions[0]))


# In[ ]:




