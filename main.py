import telebot

bot = telebot.TeleBot("8360335691:AAHWPxcarTLRdcfkUUxmPWcansyPYy6rT_w")

@bot.message_handler(commands=['start'])
def main(message):
    bot.send_message(message.chat.id, "Привет! Введи слова, которые нужны для поиска мема.")

bot.polling(none_stop=True)