import streamlit as st
from tensorflow.keras.models import load_model # type: ignore
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import numpy as np
import os

cwd = os.getcwd()
# basepath = 'NextWordPredictionProject'


with open(os.path.join(cwd,'tokenizer.pkl'), 'rb') as f:
    tokenizer = pickle.load(f)



# Prediction
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen= max_sequence_len-1, padding = 'pre')
    predicted = model.predict(token_list, verbose = 0)
    predicted_word_index = np.argmax(predicted, axis = 1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

st.title("Predicting next word with LSTM and GRU")
input_text = st.text_input('Enter the text here', 'to be or not to')
on = st.toggle("Use GRU model")
if on:
    st.write("Using GRU model!")
    # Load the model
    model = load_model(os.path.join(cwd,'next_word_gru.h5'))
else:
    st.write("Using LSTM model!")
    model = load_model(os.path.join(cwd,'next_word_lstm.h5'))

if st.button('predict next word'):
    max_len_sequence = model.input_shape[1]+1
    next_word = predict_next_word(model, tokenizer, input_text, max_len_sequence)
    st.write(f'Next word prediction : {next_word}')

