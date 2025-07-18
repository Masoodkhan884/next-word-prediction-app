import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle

# --- Load Model and Tokenizer ---
model = load_model('lstm_model.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Define max_len same as training
max_len = 10  # <-- Change to your actual `max_len` used during training


# --- Sentence Completion Function ---
def complete_sentence(seed_text, tokenizer, model, max_len, num_words=5):
    for _ in range(num_words):
        # Tokenize and pad
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_len - 1, padding='pre')

        # Predict next word
        predicted_index = np.argmax(model.predict(token_list), axis=-1)[0]

        # Get word from index
        output_word = ''
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break

        # If no word found, stop
        if output_word == '':
            break

        # Append predicted word to seed text
        seed_text += ' ' + output_word

    return seed_text


# --- Streamlit UI ---
st.title("🧠 LSTM Sentence Completion")
st.write("Enter a starting phrase, and the model will complete the sentence.")

# Input from user
user_input = st.text_input("Enter starting sentence:")

# Number of words to predict
num_words = st.slider("How many words to predict?", min_value=1, max_value=20, value=5)

# Predict button
if st.button("Complete Sentence"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        result = complete_sentence(user_input, tokenizer, model, max_len, num_words)
        st.success(f"**Completed sentence:** {result}")
