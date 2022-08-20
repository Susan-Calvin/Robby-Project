# Basic libraries
import random
import json
import tornado
import os
import telebot
import logging
from config import *


# NLP
from nltk import edit_distance

# Telegram package
from telegram import Update, ForceReply
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# Machine learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


# Set constants
RANDOM_STATE = 42
TOKEN = '5196192972:AAGYH6OP7KXiaDd4bvZZXB5PTw3iAJY7DvQ'

# Load bots vocabulary
with open('BOT_CONFIG.json', 'r', encoding="utf8") as file:
    BOT_CONFIG = json.load(file)

# Function to clean the text from symbols
def clean_text(text):
    output_text = ''
    for char in text:
        if char in "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя- ":
            output_text = output_text + char
    return output_text

# Function for locating the intent
def get_intent(input_text):
    for intent in BOT_CONFIG['intents'].keys():  # Iterate over keys in dictionary
        for example in BOT_CONFIG['intents'][intent]['examples']:
            text_a = clean_text(input_text.lower())
            text_b = clean_text(example.lower())
            if edit_distance(text_a, text_b) / max(len(text_a), len(text_b)) < 0.34:
                return intent
    return 'Not found'

# Bot function
def bot(input_text):
    intent = get_intent(input_text)
    if intent == 'Not found':
        return 'Мы вас не поняли :( Попробуйте снова'
    elif input_text == '':
        return 'Вы ничего не ввели, попробуйте снова'
    else:
        return random.choice(BOT_CONFIG["intents"][intent]["responses"])


# Vectorize sentances
vectorizer = TfidfVectorizer(ngram_range=(2, 4), analyzer='char')
X = []  ## Input messages vectors: "examples"
y = []  ## The accroding intents, features

# Test for key errors in vocabulary
for intent in BOT_CONFIG['intents'].keys():
    try:
        for example in BOT_CONFIG['intents'][intent]['examples']:
            X.append(example)
            y.append(intent)
    except KeyError:
        print(BOT_CONFIG["intents"][intent])
        print(intent)


# Preprocess and split the samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
X_vectorized = vectorizer.fit_transform(X)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Define and train the model
model = LinearSVC()
model.fit(X_train_vectorized, y_train)

def get_intent(input_text):
    return model.predict(vectorizer.transform([input_text]))[0]

def bot(input_text):
    intent = get_intent(input_text)
    return random.choice(BOT_CONFIG["intents"][intent]["responses"])

def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    update.message.reply_markdown_v2(
        fr'Hi {user.mention_markdown_v2()}\!',
        reply_markup=ForceReply(selective=True),
    )

def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')

def echo(update: Update, context: CallbackContext) -> None:
    """Echo the user message."""
    update.message.reply_text(bot(update.message.text))

def main() -> None:
    """Start the bot."""
    # Create the Updater and pass it your bots' token.
    updater = Updater(TOKEN)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))

    # on non command i.e message - echo the message on Telegram
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()

main()


