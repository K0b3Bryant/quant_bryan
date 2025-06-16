import torch
import torch.nn as nn

class SizingNN(nn.Module):
    def __init__(self, input_dim):
        super(SizingNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # Position size output between -1 and 1
        )

    def forward(self, x):
        return self.model(x)


from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def prepare_deep_learning_data(signals, returns, features_df):
    df = features_df.copy()
    df['signal'] = signals
    df['target'] = np.sign(returns.shift(-1))  # Future direction
    df.dropna(inplace=True)

    X = df.drop(columns=['target']).values
    y = df['target'].values.reshape(-1, 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    return X_tensor, y_tensor, scaler



def train_sizing_model(X, y, input_dim, epochs=50, lr=0.001):
    model = SizingNN(input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        output = model(X)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model




def predict_position_sizing(model, X):
    model.eval()
    with torch.no_grad():
        preds = model(X).squeeze().numpy()
    return preds
