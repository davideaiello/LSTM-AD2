import os
import torch
import torch.nn as nn
import arguments

args = arguments.parse_arguments()   

class AnomalyScorer:
    def __init__(self):
        super().__init__()

        self.mean = torch.tensor(0, dtype=torch.float64)
        self.var = torch.tensor(1, dtype=torch.float64)

    def forward(self, errors: torch.Tensor) -> torch.Tensor:
        mean_diff = errors - self.mean
        score = torch.mul(torch.mul(mean_diff, self.var**-1), mean_diff)
        return score.detach().cpu().numpy()


    def find_distribution(self, errors: torch.Tensor):
        self.mean = errors.mean(dim=[0, 1])
        self.var = errors.var(dim=[0, 1])

class LSTMAD(nn.Module):
    def __init__(self,
                input_size,
                lstm_layers,
                window_size,
                prediction_window_size):
        super().__init__()

        self.d = input_size
        self.lstm_layers = lstm_layers
        self.window_size = window_size
        self.l = prediction_window_size

        self.lstms = nn.LSTM(input_size=self.d, hidden_size=self.d * self.l, batch_first=True, num_layers=lstm_layers)
        self.dense = nn.Linear(in_features=self.window_size * self.d * self.l, out_features=self.d * self.l)
        self.anomaly_scorer = AnomalyScorer()

    def forward(self, x):
        x, _ = self.lstms(x)
        # x would be (b, w_l, hidden_size) -> (b, w_l, d*l)
        x = x.reshape(-1, self.window_size * self.d * self.l)
        x = self.dense(x)
        return x

    def predict(self, x, y, criterion):
        y = y.reshape(-1, self.l * self.d)
        y_hat = self.forward(x)
        loss = criterion(y_hat, y)
        return loss
    

    def save(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        path = "models/model.pth"
        torch.save({
            "model": self.state_dict(),
            "anomaly_scorer": self.anomaly_scorer
        }, path)