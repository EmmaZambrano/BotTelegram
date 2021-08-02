import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X, sample_rate = librosa.load(file_name)
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result

#Etiqueta de emociones de la base de datos RAVDESS
emotions={
  '01':'neutral',
  '02':'calmado',
  '03':'feliz',
  '04':'triste',
  '05':'enojado',
  '06':'miedo',
  '07':'disgustado',
  '08':'sorprendido'
}

#Emociones a observar
observed_emotions=['calmado', 'feliz', 'neutral', 'triste', 'enojado']

#Cargar los datos y extraer las características de cada archivo de sonido
def load_data(test_size=0.2):
    x,y=[],[]
    for file in glob.glob("C:\\Users\\zambr\\OneDrive\\Desktop\\ravdess\\Actor_*\\*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

#Dividir el conjunto de datos
x_train,x_test,y_train,y_test=load_data(test_size=0.25)

#Obtener la forma de los conjuntos de datos de entrenamiento y de prueba
print('Datos de entramiento y prueba')
print((x_train.shape[0], x_test.shape[0]))

#Obtener el número de características extraídas
print(f'Características extraídas: {x_train.shape[1]}')

#Inicializar el clasificador Perceptrón Multicapa
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=700)

#Entrenamiento del modelo
model.fit(x_train,y_train)

#Predicción para el conjunto de pruebas
y_pred=model.predict(x_test)

#Calcular la precisión del modelo
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

#Mostrar la precisión
print("Ocurrencia: {:.2f}%".format(accuracy*100))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
c= confusion_matrix(y_test, y_pred)
print('Matriz de confusión')
print(c)

modelo = 'Modelo_Análisis_Sentimientos.pkl'

with open(modelo, 'wb') as file:
    pickle.dump(model, file)


