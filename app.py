import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

# --- Load Model and Tokenizer ---
try:
    tokenizer = joblib.load("tokenizer.pkl_")
    model = load_model("lstm_model.h5")
    max_len = model.input_shape[1] + 1  # Adjust max_len based on model input

    st.set_page_config(page_title="LSTM Text Predictor", layout="centered")
    st.markdown("<h1 style='text-align: center;'>🧠 LSTM Text Generator</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # --- Text Input ---
    text_input = st.text_input("🔤 Enter a starting word or phrase:")

    if st.button("🚀 Predict Next Words"):
        if text_input.strip():
            output = text_input.strip()
            for _ in range(15):  # Predict next 15 words
                seq = tokenizer.texts_to_sequences([output])[0]
                padded = pad_sequences([seq], maxlen=max_len-1, padding='pre')
                pred_idx = np.argmax(model.predict(padded, verbose=0), axis=-1)[0]

                # Find the word from index
                next_word = ""
                for word, index in tokenizer.word_index.items():
                    if index == pred_idx:
                        next_word = word
                        break

                if next_word == "":
                    break
                output += " " + next_word

            st.success("📝 **Generated Text:**")
            st.markdown(f"```{output}```")
        else:
            st.warning("⚠️ Please enter a starting word.")
except Exception as e:
    st.error(f"❌ Error loading model or tokenizer:\n\n{e}")
