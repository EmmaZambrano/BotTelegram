import os.path
from fileinput import filename
from telegram import update
import Constants as Key
import requests, json, os
from telegram.ext import *
import librosa
import soundfile, time
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#from Emotions import extract_feature, modelo


def start(update, context):
    update.message.reply_text(f'Hola soy un bot')


def voice_handler(update: Updater, context: CallbackContext):
    supported_types = ['audio', 'voice']
    import speech_recognition as sr
    from os import path
    from pydub import AudioSegment
    try:
        message = 'Recibiendo audio'
        update.message.reply_text(message)

        # Obtener mensajes de voz de Telegram
        file = context.bot.getFile(update.message.voice.file_id)
        id = file.file_id
        filename = os.path.join('audios/', '{}.ogg'.format(id))
        file.download(filename)

        message = 'Audio guardado'
        update.message.reply_text(message)

        # files
        src = f"{filename}"
        dst = "test.wav"

        # convertir wav a mp3
        sound = AudioSegment.from_ogg(src)
        sound.export(dst, format="wav")

        message = 'Audio convertido a formato wav'
        update.message.reply_text(message)

        # Inicializar the reconocimiento de voz
        r = sr.Recognizer()

        AUDIO_FILE = "test.wav"
        with sr.AudioFile(AUDIO_FILE) as source:
            #r.adjust_for_ambient_noise(source)
            audio = r.record(source)  # Leer el archivo de audio
            text = r.recognize_google(audio, language='es-Ec')
            text = text.lower()
            message = 'Reconocimiento de voz hecho'
            update.message.reply_text(message)
            print(text)
            return text
        return -1

    except Exception as e:
        print('No se reconocio el audio')


def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X, sample_rate = librosa.load(file_name)
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result


emotions = {
    '01': 'neutral',
    '02': 'calmado',
    '03': 'feliz',
    '04': 'triste',
    '05': 'enojado',
    '06': 'miedo',
    '07': 'disgustado',
    '08': 'sorprendido'
}

# Emociones a observar
observed_emotions = ['calmado', 'feliz', 'neutral', 'triste', 'enojado']


def load_data(test_size=0.2):
    x, y = [], []

    for file in glob.glob('C:\\Users\\zambr\\PycharmProjects\\Bot\\test.wav'):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split(" ")[0]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)


file = "test.wav"
feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
x_train, x_test, y_train, y_test = load_data(test_size=0.25)
model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive',
                      max_iter=700)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
y_pre = model.predict([feature])
print(y_pre)
time.sleep(2)


def estado_animo(update: Updater):
    if y_pre[0] == "calmado":
        message = 'Su estado de animo es: Calmado'
        update.message.reply_text(message)
    elif y_pre[0] == "neutral":
        message = 'Su estado de animo es: neutral'
        update.message.reply_text(message)
    elif y_pre[0] == "feliz":
        message = 'Su estado de animo es: feliz'
        update.message.reply_text(message)
    elif y_pre[0] == "triste":
        message = 'Su estado de animo es: triste'
        update.message.reply_text(message)
    elif y_pre[0] == "enojado":
        message = 'Su estado de animo es: enojado'
        update.message.reply_text(message)
    elif y_pre[0] == "miedo":
        message = 'Su estado de animo es: miedo'
        update.message.reply_text(message)
    elif y_pre[0] == "disgustado":
        message = 'Su estado de animo es: disgustado'
        update.message.reply_text(message)
    elif y_pre[0] == "sorprendido":
        message = 'Su estado de animo es: sorprendido'
        update.message.reply_text(message)
    else:
        message = 'No se ha reconocido el audio'
        update.message.reply_text(message)


if __name__ == '__main__':
    updater = Updater(Key.API_KEY, use_context=True)

    dp = updater.dispatcher
    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(MessageHandler(Filters.voice, voice_handler))
    dp.add_handler(MessageHandler(Filters.text, estado_animo))

    updater.start_polling()
    print('Bot Listo')
    updater.idle()
