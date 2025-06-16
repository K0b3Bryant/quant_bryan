import torch
import torch.nn as nn

class SizingAdjustmentNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(SizingAdjustmentNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # Output: [-1, 1] scaled position adjustment
        )

    def forward(self, x):
        return self.model(x)



def create_lagged_features(df, window=10):
    X, y = [], []
    for i in range(window, len(df)):
        features = df.iloc[i-window:i][['signal', 'position', 'pnl']].values.flatten()
        target = df.iloc[i]['position']  # or new sizing signal
        X.append(features)
        y.append(target)
    return np.array(X), np.array(y)


from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

def train_sizing_adjuster(X, y, epochs=50, batch_size=32):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SizingAdjustmentNet(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model, scaler



def predict_adjusted_sizes(model, X, scaler):
    X_scaled = scaler.transform(X)
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        preds = model(X_tensor).squeeze().numpy()
    return preds
