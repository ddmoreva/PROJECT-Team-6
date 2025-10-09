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
    return len(set(row_tokens) & set(search_tokens))


@bot.message_handler(commands=['start'])
def start(message):
    user_id = message.chat.id
    user_data[user_id] = {'attempts': 0}
    bot.send_message(user_id, "привет! введи слова дял мема.")


@bot.message_handler(func=lambda message: True)
def process_first_query(message):
    user_id = message.chat.id
    user_data[user_id] = {
        'attempts': 1,
        'shown_indices': set()
    }

    search_tokens = custom_stem_tokenizer([message.text])[0]
    user_data[user_id]['search_tokens'] = search_tokens

    if 'tokenized_desc' not in df.columns:
        df['tokenized_desc'] = custom_stem_tokenizer(df['text'].fillna("").tolist())

    send_next_meme(user_id)


def send_next_meme(user_id):
    data = user_data[user_id]
    if data['attempts'] > 3:
        bot.send_message(user_id, "поиск завершен. попробуй другие слова")
        return

    df['matches'] = df['tokenized_desc'].apply(
        lambda x: len(set(x) & set(data['search_tokens']))
    )

    candidates = df[~df.index.isin(data['shown_indices'])]
    if candidates.empty or candidates['matches'].max() == 0:
        bot.send_message(user_id, "больше мемов подходящих не нашли((")
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


@bot.callback_query_handler(func=lambda call: True)
def handle_button(call):
    user_id = call.message.chat.id
    data = user_data.get(user_id, {})

    if call.data == "ok":
        bot.send_message(user_id, "ураааа нашли")
        user_data[user_id]['attempts'] = 0
    elif call.data == "not_ok":
        data['attempts'] += 1
        send_next_meme(user_id)


bot.polling(none_stop=True)