# KI_Tennessee_Eastman_Process

## Tim - 19.12.2023
Funktion zum Shufflen mit batch size wurde getestet und im gemeinsamen Programm implementiert.
Kleine Nachbearbeitung, sodass bei schlecht gewählter batch size keine Daten aus der ursprünglichen Liste verloren gehen.

## Paul - 17.12.2023
Habe einen ersten Versuch unternommen unsere Daten einem LSTM-Modell zu übergeben. Bissher war das jedoch nicht erfolgreich. (Funktioniert, aber definitiv kein gutes Ergebnis) 
Das shuffeln jedes Zeitpunktes ist bei Zeitreihen problematisch, da dadurch der zeitliche Zusammenhang verloren geht. 

Bei unserem gemeinsamer Datei habe ich eine Visualisierung mithilfe einer Heatmap hinzugefügt. 

## Tim - 14.12.2023
Habe keine vernünftige Shuffle-Funktion finden können, der man auch einfach eine batch-Size übergeben kann. 
Daher habe ich eine Methode geschrieben, die das kann (siehe DFF_NN_TimV2). 
Überprüft die gerne Mal und wenn sie gut ist und euch gefällt, können wir die im gemeinsamen file nutzen.

## Justin, Paul, Tim - 14.12.2023
Termin an der HSD
Ergebnis: erster Erfolg beim Training des NN, erste Auswertungen

## Justin - 09.12.2023
https://youtu.be/j-3vuBynnOE?si=JZFp1YyRqzhwhikX hab ich noch gefunden zum Thema Trainingsdatensatz erstellen 
https://youtu.be/WvoLTXIjBYU?si=QpeINelNyx-BYFI4

## Justin - 08.12.2023
Leider keine Verbesserung der Ergebnisse trotz der Tipps von Oliver Wolf. 
das zufällige trainieren der Datensätze brachte keine deutliche Verbessung! Die KI kann nach wie vor nicht zwischen den Fehlerbildern unterscheiden. Idee von paul war es sich nur einen Teil der Datensätze anzuschauen. Auch hier habe ich versucht mir einen Teil aus den Sensorwerten heraus zu suchen die Graphisch charakteristisch für den Fehler sind. Dennoch keine Verbesserung. 
Zusätzliche habe ich versucht mit Dropout Methoden und reduzieren der Trainingszeit so wie reduzieren der Neuronen und Layer das Ergebnis zu verbessern. Auch hier blieb der Erfolg aus. 


