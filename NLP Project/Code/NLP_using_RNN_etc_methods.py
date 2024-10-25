
import numpy as np
import pandas as pd

import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LayerNormalization, Dropout, Layer, Embedding, Input, GlobalAveragePooling1D, Dense, Flatten, SimpleRNN, GlobalMaxPooling1D,  LSTM, SpatialDropout1D, Activation, MultiHeadAttention

from keras.models import Sequential, Model
from keras.callbacks import  EarlyStopping
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


X_trn_data = pd.read_csv('./project2_training_data.txt', encoding="utf8",na_filter=False, on_bad_lines='skip')
y_trn= pd.read_csv('./project2_training_data_labels.txt', encoding="utf8", index_col=False)

import csv,sys
fl=open('./project2_training_data.txt',encoding="utf8")  # Source: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
reader = list(csv.reader(fl,delimiter='\n'))
f2 = open('./project2_training_data_labels.txt',encoding="utf8")
reader2 = list(csv.reader(f2,delimiter='\n'))
# Load test data
f3=open('./project2_test_data.txt',encoding="utf8")  # Source: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
reader3 = list(csv.reader(f3,delimiter='\n'))


test = []
for i in reader3:
    test.append(i[0])
data=[]; labels=[];
for i,item in enumerate(reader2):
    if item[0]=='positive':
        labels.append(0)
    elif item[0] == 'negative':
        labels.append(1)
    else:
        labels.append(2)
    data.append(reader[i][0])

X_trn = data
y_trn = labels
X_test = test

vocab_size = 1000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each sequence
embed_dim = 200


# tokenizer
tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
# fit tokenizer
tokenizer.fit_on_texts(X_trn)
tokenizer.fit_on_texts(X_test)
# text to sequences
train_texts_to_int = tokenizer.texts_to_sequences(X_trn)   # list type; each element in this list have seq. nos. of all words 
test_texts_to_int = tokenizer.texts_to_sequences(X_test)


train_int_texts_to_pad = tf.keras.preprocessing.sequence.pad_sequences(train_texts_to_int, maxlen=maxlen)
test_int_texts_to_pad = tf.keras.preprocessing.sequence.pad_sequences(test_texts_to_int, maxlen=maxlen)


# x_train, x_valid, x_test
x_train = train_int_texts_to_pad
x_test = test_int_texts_to_pad



# Check Total Vocab Size
total_vocab_size = len(tokenizer.word_index) + 1
print('Total Vocabulary Size (Untrimmed): %d' % total_vocab_size)
print('Vocabulary Size (trimmed): %d' % vocab_size)

Y = pd.get_dummies(y_trn).values
#Splitting Train / Valid
X_train, X_valid, Y_train, Y_valid = train_test_split(x_train,Y, test_size = 0.20, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_valid.shape,Y_valid.shape)
Y

"""Transformer"""

# Transformer block
class TransformerBlock(Layer):
    # initialization of various layers
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim) # multi-head attention layer
        # feed-forward network
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), 
             Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)  # batch normalization
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)   # dropout layers
        self.dropout2 = Dropout(rate)
    
    # defined the layers according to the architecture of the transformer block
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)  # multi-head attention layer
        attn_output = self.dropout1(attn_output, training=training)  # 1st dropout
        out1 = self.layernorm1(inputs + attn_output)  # 1st normalization
        ffn_output = self.ffn(out1)   # feed-forward network
        ffn_output = self.dropout2(ffn_output, training=training)  # 2nd dropout
        out2 = self.layernorm2(out1 + ffn_output)  # 2nd normalization
        return out2

# positional embeddings
class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim) # embedding layer for tokens
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)  # embedding layer for token positions

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

# Embedding size for each token 200
num_heads = 8  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block1 = TransformerBlock(embed_dim, num_heads, ff_dim)
transformer_block2 = TransformerBlock(embed_dim, num_heads, ff_dim)
transformer_block3 = TransformerBlock(embed_dim, num_heads, ff_dim)

x = transformer_block1(training = "auto", inputs = x)
x = transformer_block2(training = "auto", inputs = x)
x = transformer_block3(training = "auto", inputs = x)
x = GlobalAveragePooling1D()(x)


x = Dropout(0.1)(x)
x = Dense(32, activation="relu")(x)
x = Dropout(0.1)(x)
outputs = Dense(3, activation="sigmoid")(x)

model = Model(inputs=inputs, outputs=outputs)
model.summary() # print model summary

model.compile(optimizer="RMSprop", loss="binary_crossentropy", metrics=["accuracy",'Precision','Recall'])
history = model.fit(X_train, Y_train, 
                    batch_size=4096//16, epochs=100, 
                    validation_split=0.2,
                   callbacks=[EarlyStopping(monitor='val_loss', patience=3,verbose = 1)]
                   )

result_trsfmr = model.evaluate(X_valid,Y_valid,batch_size = 512)
print(f'Test set FFNN MODEL\n  Loss: {result_trsfmr[0]:0.3f}\n  Accuracy: {result_trsfmr[1]:0.3f}')

"""FFNN"""

ffnn = Sequential() 
ffnn.add(Embedding(vocab_size,embed_dim, input_length=maxlen))
ffnn.add(Flatten()) 
ffnn.add(Dense(8, activation='softplus', name='Hidden-Layer')) 
ffnn.add(Dense(16, activation='relu')) 
ffnn.add(Dense(16, activation='relu'))
ffnn.add(Dense(3, activation='softmax', name='Output-Layer'))

## Compile the keras model
ffnn.compile(optimizer='RMSprop',
              loss='binary_crossentropy',
              metrics=['Accuracy', 'Precision', 'Recall']
              )
ffnn.summary()

## Fit keras model on the dataset
ffnn.fit(X_train, Y_train, 
          batch_size=32,
          epochs=100,
          callbacks=[EarlyStopping(monitor='val_loss', patience=5,verbose = 1)],validation_split = 0.2)

result_ffnlm = ffnn.evaluate(X_valid,Y_valid,batch_size = 32)
print(f'Test set FFNN MODEL\n  Loss: {result_ffnlm[0]:0.3f}\n  Accuracy: {result_ffnlm[1]:0.3f}')

"""Stacked LSTM"""

Stacklstm = Sequential()
Stacklstm.add(Embedding(vocab_size,embed_dim, input_length=maxlen))
Stacklstm.add(LSTM(8, return_sequences=True))
Stacklstm.add(LSTM(32, return_sequences=True)) 
Stacklstm.add(LSTM(64, return_sequences=True)) 
Stacklstm.add(LSTM(16)) 
Stacklstm.add(Dense(3, activation='sigmoid'))
Stacklstm.compile(optimizer='RMSprop',
              loss='binary_crossentropy',
              metrics=['Accuracy', 'Precision', 'Recall']
              )
Stacklstm.summary()

## Fit keras model on the dataset
Stacklstm.fit(X_train, Y_train, 
          batch_size=64,
          epochs=100
          ,callbacks=[EarlyStopping(monitor='val_loss', patience=5,verbose = 1)],validation_split = 0.1)

result_lstm = Stacklstm.evaluate(X_valid,Y_valid,batch_size = 1024)
print(f'Test set Stacked LSTM MODEL\n  Loss: {result_lstm[0]:0.3f}\n  Accuracy: {result_lstm[1]:0.3f}')

"""Stack RNN"""

stackrnn = Sequential()
stackrnn.add(Embedding(vocab_size,embed_dim, input_length=maxlen))
stackrnn.add(SimpleRNN(8, return_sequences=True,unroll=True)) 
stackrnn.add(SimpleRNN(32, return_sequences=True,unroll=True))
stackrnn.add(SimpleRNN(16, return_sequences=True,unroll=True)) 
stackrnn.add(SimpleRNN(8,unroll=True)) 
stackrnn.add(Dense(3, activation='softmax'))
stackrnn.compile(optimizer='RMSprop',
              loss='binary_crossentropy',
              metrics=['Accuracy', 'Precision', 'Recall']
              )
stackrnn.summary()

## Fit keras model on the dataset
stackrnn.fit(X_train, Y_train, 
          batch_size=4096,
          epochs=100
          ,callbacks=[EarlyStopping(monitor='val_loss', patience=5,verbose = 1)],validation_split = 0.1)

result_rnn = stackrnn.evaluate(X_valid,Y_valid,batch_size = 512)
print(f'Test set Stack RNN MODEL\n  Loss: {result_rnn[0]:0.3f}\n  Accuracy: {result_rnn[1]:0.3f}')

np.argmax(ffnn.predict(X_valid),axis=1)

np.argmax(model.predict(X_valid),axis=1)

t = np.argmax(Stacklstm.predict(x_test),axis=1)

pred = np.array(list(t))

def show(predicted):
    R = []
    for i in range(0,len(predicted)):
        if predicted[i] == 0:
            R.append("positive")
        elif predicted[i] == 1:
            R.append("negative")
        elif predicted[i] == 2:
            R.append("neutral")
    return R

prediction = pd.DataFrame(show(pred), columns=['predictions']).to_csv('./prediction.csv', encoding="utf8", header=False, index=False)

