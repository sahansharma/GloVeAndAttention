import numpy as np
import math
import tensorflow as tf
import pickle
import string
import random

# Configuration
filename = 'glove.6B.50d.txt'
word_vec_dim = 50
h = 8  # Number of attention heads
N = 1  # Number of encoder/decoder layers
dqkv = 32  # Dimension of Q/K/V
d_ff = 1024  # Feed-forward dimension
batch_size = 64
learning_rate = 0.001
epochs = 200
keep_prob = 0.9

# Load embeddings and data
def load_embeddings(filename):
    vocab = []
    embd = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            vocab.append(parts[0])
            embd.append([float(x) for x in parts[1:]])
    return vocab, np.array(embd, dtype=np.float32)

vocab, embd = load_embeddings(filename)

# Add special tokens
special_tokens = ['<UNK>', '<EOS>', '<PAD>']
for token in special_tokens:
    if token not in vocab:
        vocab.append(token)
        embd = np.vstack([embd, np.random.normal(size=word_vec_dim)])

# Load preprocessed data
with open('AmazonPICKLE', 'rb') as fp:
    data = pickle.load(fp)
    
vocab_limit = data[0]
max_len = len(data[3][0][0])
vocab_len = len(vocab_limit)
embd_limit = np.array([word2vec(w) for w in vocab_limit], dtype=np.float32)

# Precompute positional encodings
def positional_encoding(max_len, d_model):
    pos = np.arange(max_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    pe = pos * angle_rates
    pe[:, 0::2] = np.sin(pe[:, 0::2])
    pe[:, 1::2] = np.cos(pe[:, 1::2])
    return pe.astype(np.float32)

pe_encoder = positional_encoding(2048, word_vec_dim)
pe_decoder = positional_encoding(max_len, word_vec_dim)

# Transformer Model Components
class Transformer(tf.keras.Model):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder_layers = [EncoderLayer() for _ in range(N)]
        self.decoder_layers = [DecoderLayer() for _ in range(N)]
        self.final_layer = tf.keras.layers.Dense(vocab_len)
        
    def call(self, inputs, training=True):
        src, tgt = inputs
        enc_output = src
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, training=training)
            
        dec_output = tgt
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, training=training)
            
        return self.final_layer(dec_output)

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention()
        self.ffn = FeedForward()
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(1 - keep_prob)
        self.dropout2 = tf.keras.layers.Dropout(1 - keep_prob)
        
    def call(self, x, training):
        attn_output = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention()
        self.mha2 = MultiHeadAttention()
        self.ffn = FeedForward()
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(1 - keep_prob)
        self.dropout2 = tf.keras.layers.Dropout(1 - keep_prob)
        self.dropout3 = tf.keras.layers.Dropout(1 - keep_prob)
        
    def call(self, x, enc_output, training):
        attn1 = self.mha1(x, x, x, look_ahead_mask=True)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        
        attn2 = self.mha2(out1, enc_output, enc_output)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        return self.layernorm3(ffn_output + out2)

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.wq = tf.keras.layers.Dense(dqkv * h)
        self.wk = tf.keras.layers.Dense(dqkv * h)
        self.wv = tf.keras.layers.Dense(dqkv * h)
        self.dense = tf.keras.layers.Dense(word_vec_dim)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, h, dqkv))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, q, k, v, look_ahead_mask=False):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, look_ahead_mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, h * dqkv))
        output = self.dense(concat_attention)
        return output

def scaled_dot_product_attention(q, k, v, mask=False):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    if mask:
        look_ahead_mask = tf.linalg.band_part(
            tf.ones_like(scaled_attention_logits), -1, 0)
        scaled_attention_logits += (look_ahead_mask * -1e9)
        
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights

class FeedForward(tf.keras.layers.Layer):
    def __init__(self):
        super(FeedForward, self).__init__()
        self.dense1 = tf.keras.layers.Dense(d_ff, activation='relu')
        self.dense2 = tf.keras.layers.Dense(word_vec_dim)
        
    def call(self, x):
        x = self.dense1(x)
        return self.dense2(x)

# Training Setup
optimizer = tf.keras.optimizers.Adam(learning_rate)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def loss_function(real, pred, padding_mask):
    loss = loss_object(real, pred)
    mask = tf.math.logical_not(tf.math.equal(real, vocab_limit.index('<PAD>')))
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

# Dataset Preparation
def create_dataset(batches_x, batches_y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((batches_x, batches_y))
    dataset = dataset.shuffle(len(batches_x)).batch(batch_size)
    return dataset.prefetch(tf.data.AUTOTUNE)

# Training Loop
def train_step(model, inputs, targets, pe_enc, pe_dec):
    with tf.GradientTape() as tape:
        predictions = model([inputs + pe_enc, targets + pe_dec], training=True)
        loss = loss_function(targets, predictions, None)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Initialize and Train
model = Transformer()
train_data = create_dataset(train_batches_x, train_batches_y, batch_size)

for epoch in range(epochs):
    total_loss = 0
    for batch, (inputs, targets) in enumerate(train_data):
        inputs_pe = inputs + pe_encoder[:tf.shape(inputs)[1]]
        targets_pe = targets + pe_decoder[:tf.shape(targets)[1]]
        loss = train_step(model, inputs_pe, targets_pe)
        total_loss += loss
        
    print(f'Epoch {epoch+1} Loss {total_loss / (batch+1)}')

# Save Model
model.save_weights('transformer_model.h5')