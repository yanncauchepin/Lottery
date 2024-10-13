import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, GlobalAveragePooling1D
from tensorflow.keras.layers import MultiHeadAttention, Add
from tensorflow.keras.optimizers import Adam

root_path = "/home/yanncauchepin/Git/Lottery"

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    attn_output = Dropout(dropout)(attn_output)
    out1 = Add()([inputs, attn_output])
    out1 = LayerNormalization(epsilon=1e-6)(out1)

    ffn_output = Dense(ff_dim, activation="relu")(out1)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)
    ffn_output = Dropout(dropout)(ffn_output)
    out2 = Add()([out1, ffn_output])
    out2 = LayerNormalization(epsilon=1e-6)(out2)
    return out2

def build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    inputs = Input(shape=input_shape)

    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D()(x)

    x = Dense(mlp_units, activation="relu")(x)
    x = Dropout(mlp_dropout)(x)
    x = Dense(input_shape[-1], activation="sigmoid")(x)

    model = Model(inputs, x)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def prepare_data_transformer(df, window_size):
    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(df[i:i + window_size])
        y.append(df[i + window_size])
    return np.array(X), np.array(y)

def binary_crossentropy(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-12, 1.0)
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred), axis=1))

def meta_modeling(lottery, df, size, numbers):
    files = glob.glob(os.path.join(root_path, f'history_transformer/{lottery}/*'))
    for file in files:
        if os.path.exists(file):
            os.remove(file)

    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df)

    window_size = 500

    X, y = prepare_data_transformer(df_scaled, window_size)

    # Build Transformer model
    model = build_transformer_model(input_shape=(X.shape[1], X.shape[2]), head_size=128, num_heads=4, ff_dim=128, num_transformer_blocks=2, mlp_units=128, dropout=0.1, mlp_dropout=0.1)

    # Train the model
    model.fit(X, y, epochs=10, batch_size=64, verbose=2)

    # Predict and plot
    i = 0
    for date, draw in df.iloc[-50:].iterrows():
        X_sample = X[i].reshape(1, X.shape[1], X.shape[2])  # Input is a single sample
        i += 1
        predicted_proba = model.predict(X_sample)[0]  # Get predicted probabilities

        # Inverse transform to get actual draw values
        predicted_proba_rescaled = scaler.inverse_transform([predicted_proba])[0]  # Adjusted here

        print(f'{date}: {binary_crossentropy(draw, predicted_proba_rescaled)}')

        # Plot the actual vs predicted
        plt.figure()
        plt.bar(range(len(draw)), draw * 2 * np.max(predicted_proba_rescaled)) 
        plt.bar(range(len(predicted_proba_rescaled)), predicted_proba_rescaled)
        os.makedirs(os.path.join(root_path, f'history_transformer/{lottery}'), exist_ok=True) 
        plt.savefig(os.path.join(root_path, f'history_transformer/{lottery}/{date}.png'))
        plt.close()

    next_draw_proba = model.predict(X[-1].reshape(1, X.shape[1], X.shape[2]))[0]  # Adjusted for last input
    next_draw_proba_rescaled = scaler.inverse_transform([next_draw_proba])[0]  # Adjusted for inverse transform

    proba_df = pd.DataFrame(next_draw_proba_rescaled, index=range(1, numbers + 1))
    proba_df = proba_df.sort_values(by=0, ascending=False)

    return proba_df

if __name__ == '__main__':
    df = pd.read_csv('data/all_concat_one_hot_ball_loto.csv', index_col=0)
    result = meta_modeling("loto_ball", df, 5, 49)
    print(result)
