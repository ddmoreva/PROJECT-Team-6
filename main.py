import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from string import punctuation
from nltk.tokenize import TweetTokenizer
from PIL import Image
import io

try:
    df = pd.read_parquet("hf://datasets/foldl/rumeme-desc/data/train-00000-of-00001.parquet")
except Exception as e:
    print(f"Ошибка загрузки данных: {e}")
    df = pd.DataFrame()

tw = TweetTokenizer()
stemmer = SnowballStemmer("russian")
bot = telebot.TeleBot("8360335691:AAHWPxcarTLRdcfkUUxmPWcansyPYy6rT_w")
user_data = {}
nltk.download('stopwords', quiet=True)
stop = stopwords.words("russian")


def custom_stem_tokenizer(text):
    """
    Обрабатывает текст, приводит к нижнему регистру, токенизирует, удаляет стоп-слова и пунктуацию,
    применяет стемминг.

    Параметры:
        text (list): Список строк (описаний мемов или запросов)

    Возвращает:
        list: Список списков — обработанных токенов для каждого описания
    """
    transformed = []
    for description in text:
        description = description.lower()
        tokenized = tw.tokenize(description)
        tokens = [stemmer.stem(token) for token in tokenized
                  if token not in stop
                  and token not in punctuation
                  and not (len(token) == 1 and ord(token) >= 128)]
        transformed.append(tokens)
    return transformed


def count_matches(row_tokens, search_tokens):
    """
    Считает количество совпадающих слов между описанием мема и запросом пользователя.

    Параметры:
        row_tokens (list): Слова из описания мема
        search_tokens (list): Слова из запроса пользователя

    Возвращает:
        int: Количество общих слов
    """
    return len(set(row_tokens).intersection(set(search_tokens)))


def get_main_menu():
    """
    Создаёт клавиатуру с двумя кнопками.

    Возвращает:
        InlineKeyboardMarkup: Готовая клавиатура с кнопками.
    """
    markup = InlineKeyboardMarkup()
    btn_start = InlineKeyboardButton("🚀 начинаем поиск!", callback_data="start_search")
    btn_help = InlineKeyboardButton("🆘 помогите, не могу разобраться с ботом", url="https://t.me/ksujpg")
    markup.add(btn_start)
    markup.add(btn_help)
    return markup


def send_next_meme(user_id):
    """
    Находит и отправляет следующий подходящий мем пользователю.
    - Сравнивает слова из запроса и описания мема
    - Исключает уже показанные мемы
    - Ограничивает попытки (макс. 3)
    - Если мемов нет — предлагает начать новый поиск

    Параметры:
        user_id (int): ID пользователя в Telegram
    """
    if user_id not in user_data:
        user_data[user_id] = {
            'attempts': 1,
            'shown_indices': set(),
            'finished': False,
            'search_tokens': []
        }

    data = user_data[user_id]
    if data['attempts'] > 3:
        bot.send_message(user_id, "поиск завершен. попробуй другие слова")
        return

    df['matches'] = df['stan'].apply(
        lambda x: len(set(x).intersection(set(data['search_tokens'])))
    )

    candidates = df[~df.index.isin(data['shown_indices'])]
    if candidates.empty or candidates['matches'].max() == 0:
        bot.send_message(user_id, "поиск завершен.")
        user_data.pop(user_id, None)
        markup = get_main_menu()
        bot.send_message(user_id, "попробуй другие слова!", reply_markup=markup)
        return

    best_idx = candidates['matches'].idxmax()
    row = df.loc[best_idx]
    image_bytes = row['image']['bytes']
    image = Image.open(io.BytesIO(image_bytes))
    img_io = io.BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)
    markup = InlineKeyboardMarkup()
    markup.add(
        InlineKeyboardButton("ок", callback_data="ok"),
        InlineKeyboardButton("не ок", callback_data="not_ok")
    )
    bot.send_photo(user_id, img_io, caption="вот такой мем, подходит тебе?", reply_markup=markup)
    data['shown_indices'].add(best_idx)


def has_active_search(user_id) -> bool:
    """
    Проверяет, идёт ли у пользователя активный поиск мема.

    Параметры:
        user_id (int): ID пользователя

    Возвращает:
        bool: True — если поиск ещё не завершён и попытки < 4
    """
    if user_id not in user_data:
        return False

    data = user_data[user_id]
    return (
            data.get('attempts', 0) < 4
            and not data.get('finished', True)
    )


def start_new_search(user_id, query_text):
    """
    Запускает новый поиск мема.

    Параметры:
        user_id (int): ID пользователя
        query_text (str): Текст, введённый пользователем
    """
    user_data[user_id] = {
        'attempts': 1,
        'shown_indices': set(),
        'finished': False
    }

    try:
        search_tokens = custom_stem_tokenizer([query_text])[0]
        user_data[user_id]['search_tokens'] = search_tokens
    except:
        bot.send_message(user_id, "ошибка обработки запроса. попробуй ещё раз.")
        return

    if 'stan' not in df.columns:
        df['stan'] = custom_stem_tokenizer(df['text'].fillna("").tolist())

    send_next_meme(user_id)


@bot.message_handler(commands=['start'])
def start(message):
    """
    Обрабатывает команду /start.

    Параметры:
        message: Сообщение от пользователя
    """
    user_id = message.chat.id
    user_data[user_id] = {'attempts': 0}
    markup = get_main_menu()
    bot.send_message(
        user_id,
        "привет! я помогу найти мем по твоим словам.",
        reply_markup=markup
    )


@bot.callback_query_handler(func=lambda call: call.data == "start_search")
def handle_start_button(call):
    """
    Обрабатывает нажатие кнопки "начинаем поиск!".

    Параметры:
        call: Нажатие на кнопку
    """
    user_id = call.message.chat.id

    try:
        bot.delete_message(chat_id=user_id, message_id=user_id)
    except:
        pass

    bot.send_message(user_id, "супер! введи слова для поиска мема.")


@bot.callback_query_handler(func=lambda call: call.data in ["ok", "not_ok"])
def handle_button(call):
    """
    Обрабатывает нажатие кнопок "ок" / "не ок".

    Параметры:
        call: Нажатие на кнопку
    """
    user_id = call.message.chat.id
    data = user_data.get(user_id, {})

    if call.data == "ok":
        bot.send_message(user_id, "ураааа нашли 🎉")
        user_data[user_id]['attempts'] = 0
        bot.delete_message(chat_id=user_id, message_id=user_id)
        markup = get_main_menu()
        bot.send_message(user_id, "хочешь найти ещё один мем?", reply_markup=markup)

    elif call.data == "not_ok":
        data['attempts'] += 1
        bot.delete_message(chat_id=call.message.chat.id, message_id=call.message.message_id)
        send_next_meme(user_id)


@bot.message_handler(func=lambda message: True)
def process_first_query(message):
    """
    Обрабатывает введённый пользователем текст.

    Параметры:
        message: Сообщение от пользователя
    """
    user_id = message.chat.id

    if message.content_type != 'text':
        bot.send_message(
            user_id,
            "плиз, вводи только текст для поиска мема. мы еще не слишком хороши, чтобы обрабатывать другого типа запросы."
        )

        markup = get_main_menu()
        bot.send_message(user_id, "ну че, приступим?", reply_markup=markup)
        return

    if message.text.startswith('/'):
        return

    if has_active_search(user_id):
        markup = InlineKeyboardMarkup()
        markup.add(
            InlineKeyboardButton("начинаем по новой", callback_data="new_search"),
            InlineKeyboardButton("извините, продолжаем искать", callback_data="continue_search")
        )
        bot.send_message(
            user_id,
            "у тебя уже идёт поиск! что хочешь сделать?",
            reply_markup=markup
        )
        return

    start_new_search(user_id, message.text)


@bot.callback_query_handler(func=lambda call: call.data in ["new_search", "continue_search"])
def handle_search_choice(call):
    """
    Обрабатывает выбор: начать новый поиск или продолжить старый.

    Параметры:
        call: Нажатие на кнопку
    """
    user_id = call.message.chat.id

    try:
        bot.delete_message(chat_id=call.message.chat.id, message_id=call.message.message_id)
    except:
        pass

    if call.data == "new_search":
        bot.send_message(user_id, "окей! введи новые слова для поиска мема.")
        user_data[user_id]['awaiting_new_query'] = True

    elif call.data == "continue_search":
        bot.send_message(user_id, "продолжаем старый поиск!")
        send_next_meme(user_id)


bot.polling(none_stop=True)