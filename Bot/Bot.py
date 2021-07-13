from telegram.ext import Updater, CommandHandler

def start(update, context):
    update.message.reply_text('Hola, Humano')

if __name__ == '__main__':
    updater = Updater(token= '1857138719:AAH4KE2Pk9GrRzd9tW0ERypTxtDzWJnm5KI', use_context=True)

    dp = updater.dispatcher
    dp.add_handler(CommandHandler('start', start))

    updater.start_polling()
    updater.idle()