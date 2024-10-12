import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout
from tensorflow.keras.layers import MultiHeadAttention, Add, TimeDistributed
from tensorflow.keras.optimizers import Adam

root_path = "/home/yanncauchepin/Git/Lottery"

def create_padding_mask(x):
    mask = np.where(x == 0, 1, 0)
    return mask[:, np.newaxis, np.newaxis, :]

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Multi-head self-attention
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    attn_output = Dropout(dropout)(attn_output)
    out1 = Add()([inputs, attn_output])
    out1 = LayerNormalization(epsilon=1e-6)(out1)

    # Feed Forward layer
    ffn_output = Dense(ff_dim, activation="relu")(out1)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)
    ffn_output = Dropout(dropout)(ffn_output)
    out2 = Add()([out1, ffn_output])
    out2 = LayerNormalization(epsilon=1e-6)(out2)
    return out2

def build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    inputs = Input(shape=input_shape)

    # Positional encoding could be added if we want to provide positional information
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = TimeDistributed(Dense(mlp_units, activation="relu"))(x)
    x = Dropout(mlp_dropout)(x)
    x = TimeDistributed(Dense(input_shape[-1], activation="softmax"))(x)

    model = Model(inputs, x)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def prepare_data_transformer(df, window_size):
    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(df[i:i+window_size])
        y.append(df[i+1:i+window_size+1])
    return np.array(X), np.array(y)

def binary_crossentropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-12, 1.0)
    cross_entropy = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return cross_entropy

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
    model.fit(X, y, epochs=2, batch_size=64, verbose=2)

    # Predict and plot
    i = 0
    for date, draw in df.iloc[-50:].iterrows():
        if i >= len(X):
            break
        i += 1
        X_sample = X[i].reshape(1, X.shape[1], X.shape[2])  # Input is a single sample

        predicted_proba = model.predict(X_sample)[0]  # Get predicted probabilities

        # Inverse transform to get actual draw values
        predicted_proba_rescaled = scaler.inverse_transform([predicted_proba[0]])[0]

        print(f'{date}: {binary_crossentropy(draw, predicted_proba_rescaled)}')

        # Plot the actual vs predicted
        plt.figure()
        plt.bar(range(len(draw)), draw * 2 * np.max(predicted_proba_rescaled)) 
        plt.bar(range(len(predicted_proba_rescaled)), predicted_proba_rescaled)
        os.makedirs(os.path.join(root_path, f'history_transformer/{lottery}'), exist_ok=True) 
        plt.savefig(os.path.join(root_path, f'history_transformer/{lottery}/{date}.png'))
        plt.close()
    # After prediction, return the predicted probabilities for the next draw
    next_draw_proba = model.predict(y[-1].reshape(1, X.shape[1], X.shape[2]))[0]
    next_draw_proba_rescaled = scaler.inverse_transform([next_draw_proba[0]])[0]

    proba_df = pd.DataFrame(next_draw_proba_rescaled, index=range(1, numbers + 1))
    proba_df = proba_df.sort_values(by=0, ascending=False)

    return proba_df

if __name__ == '__main__':
    df = pd.read_csv('data/all_concat_one_hot_ball_euromillions.csv', index_col=0)
    result = meta_modeling("euromillions_ball", df, 5, 50)
    print(result)