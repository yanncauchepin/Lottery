import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import glob
import shutil
from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent

def prepare_data_transformer(df, window_size, lottery):
    save_path = Path(ROOT_PATH, f'transformer_data/{lottery}')
    if os.path.exists(save_path) and os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    for i in range(len(df) - window_size):
        X_sample = df.iloc[i:i + window_size,].to_numpy()
        y_sample = df.iloc[i + window_size,].to_numpy()
        torch.save((torch.tensor(X_sample, dtype=torch.float32), torch.tensor(y_sample, dtype=torch.float32)),
                   os.path.join(save_path, f'data_{i}.pt'))
    
    print(f"Data prepared and saved to {save_path}")
    return save_path

class LotteryDataset(Dataset):
    def __init__(self, data_path):
        self.files = glob.glob(os.path.join(data_path, '*.pt'))
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        X, y = torch.load(self.files[idx])
        return X, y

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, head_size, num_heads, ff_dim, dropout=0.0):
        super(TransformerEncoder, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, input_dim),
        )
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0.0):
        super(TransformerModel, self).__init__()

        if input_shape[1] % num_heads != 0:
            print(f"Adjusting num_heads from {num_heads} to fit input_dim {input_shape[1]}")
            num_heads = max(1, input_shape[1] // head_size)

        self.layers = nn.ModuleList([
            TransformerEncoder(input_shape[1], head_size, num_heads, ff_dim, dropout)
            for _ in range(num_transformer_blocks)
        ])
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_shape[1], mlp_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_units, input_shape[1]),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.pool(x.permute(0, 2, 1)).squeeze()
        return self.fc(x)

def meta_modeling(lottery, df, size, numbers):
    window_size = 500
    batch_size = 3
    data_path = prepare_data_transformer(df, window_size, lottery)
    dataset = LotteryDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_shape = (500, df.shape[1])
    
    model = TransformerModel(
        input_shape=input_shape,
        head_size=128,
        num_heads=4,
        ff_dim=128,
        num_transformer_blocks=2,
        mlp_units=128,
        dropout=0.1
    ).to('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(10):
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to('cuda'), y_batch.to('cuda')
            optimizer.zero_grad()
            outputs = model(X_batch)
            outputs = outputs.view_as(y_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        for i, (date, draw) in enumerate(df.iloc[-50:,].iterrows()):
            X_sample = df.iloc[-(window_size+i+1):-(i+1),].to_numpy().reshape(1, window_size, df.shape[1])
            X_sample = torch.tensor(X_sample, dtype=torch.float32).to('cuda')
            predicted_proba = model(X_sample).cpu().numpy().flatten()

            print(f'{date}: {criterion(torch.tensor(draw).float(), torch.tensor(predicted_proba)).item()}')

            plt.figure()
            plt.bar(range(len(draw)), draw * 2 * np.max(predicted_proba))
            plt.bar(range(len(predicted_proba)), predicted_proba)
            os.makedirs(Path(ROOT_PATH, f'history_transformer/{lottery}'), exist_ok=True)
            plt.savefig(Path(ROOT_PATH, f'history_transformer/{lottery}/{date}.png'))
            plt.close()

    next_draw_proba = predicted_proba
    proba_df = pd.DataFrame(next_draw_proba, index=range(1, numbers + 1), columns=["Probability"]).sort_values(by="Probability", ascending=False)
    
    return proba_df

if __name__ == '__main__':
    df = pd.read_csv('data/all_concat_one_hot_ball_euromillions.csv', index_col=0)
    result = meta_modeling("euromillions_ball", df, 5, 50)
    print(result)