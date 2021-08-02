import os.path
from telegram import update
import Constants as Key
import requests, json, os
from telegram.ext import *
from nltk import *
from textblob import *
#import librosa
#import soundfile, time
#import os, glob, pickle
#import numpy as np
#from sklearn.model_selection import train_test_split
#from sklearn.neural_network import MLPClassifier
#from sklearn.metrics import accuracy_score



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

        #message = 'Audio convertido a formato wav'
        #update.message.reply_text(message)

        # Inicializar the reconocimiento de voz
        r = sr.Recognizer()

        AUDIO_FILE = "test.wav"
        with sr.AudioFile(AUDIO_FILE) as source:
            #r.adjust_for_ambient_noise(source)
            audio = r.record(source)  # Leer el archivo de audio
            text = r.recognize_google(audio, language='es-Ec')
            text = text.lower()
            #message = 'Reconocimiento de voz hecho'
            #update.message.reply_text(message)
            tb = TextBlob(text).translate(from_lang='es', to='en')
            print(text)
            print(tb.sentiment)
            if tb.sentiment.polarity > 0.4:
                message = 'Sentimiento positivo'
                update.message.reply_text(message)
                #print('Sentimiento positivo')
            elif tb.sentiment.polarity == 0:
                message = 'Sentimiento neutral'
                update.message.reply_text(message)
                #print('Sentimiento neutral')
            else:
                message = 'Sentimiento negativo'
                update.message.reply_text(message)
                #print('Sentimiento negativo')
            return text
        return result
        return -1

    except Exception as e:
        print('No se reconocio el audio')

if __name__ == '__main__':
    updater = Updater(Key.API_KEY, use_context=True)

    dp = updater.dispatcher
    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(MessageHandler(Filters.voice, voice_handler))
    dp.add_handler(MessageHandler(Filters.text, estado_animo))

    updater.start_polling()
    print('Bot Listo')
    updater.idle()
