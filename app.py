import streamlit as st
import torch
import numpy as np
import pickle
import re
from nltk.corpus import stopwords
import torch.nn as nn

# Load the vocabulary
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)


# Define the SentimentRNN model class
class SentimentRNN(nn.Module):
    def __init__(self, no_layers, vocab_size, hidden_dim, embedding_dim, drop_prob=0.5):
        super(SentimentRNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.no_layers = no_layers

        # Embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dim,
                            num_layers=no_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.hidden_dim, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        sig_out = self.sig(out)
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]
        return sig_out, hidden

    def init_hidden(self, batch_size):
        h0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(device)
        c0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(device)
        hidden = (h0, c0)
        return hidden


# Model parameters (use the values you trained with)
no_layers = 2
vocab_size = len(vocab) + 1  # extra 1 for padding
embedding_dim = 64
output_dim = 1
hidden_dim = 256

# Instantiate the model and load the state dict
model = SentimentRNN(no_layers, vocab_size, hidden_dim, embedding_dim, drop_prob=0.5)
# Load the model state dict with map_location to CPU
model.load_state_dict(torch.load('state_dict.pt', map_location=torch.device('cpu')))

model.eval()

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Function to preprocess and predict sentiment remains the same
def preprocess_string(s):
    s = re.sub(r"[^\w\s]", '', s)
    s = re.sub(r"\s+", '', s)
    s = re.sub(r"\d", '', s)
    return s


def predict_text(text):
    word_seq = np.array([vocab[preprocess_string(word)] for word in text.split()
                         if preprocess_string(word) in vocab.keys()])
    word_seq = np.expand_dims(word_seq, axis=0)
    pad = torch.from_numpy(padding_(word_seq, 500))
    inputs = pad.to(device)
    batch_size = 1
    h = model.init_hidden(batch_size)
    h = tuple([each.data for each in h])
    output, h = model(inputs, h)
    return output.item()


def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features


# Streamlit UI
st.title("Sentiment Analysis")
user_input = st.text_area("Enter a movie review:")

if st.button("Predict"):
    sentiment_score = predict_text(user_input)
    status = "positive" if sentiment_score > 0.5 else "negative"
    sentiment_score = (1 - sentiment_score) if status == "negative" else sentiment_score
    st.write(f'Predicted sentiment is {status} with a probability of {sentiment_score:.2f}')
