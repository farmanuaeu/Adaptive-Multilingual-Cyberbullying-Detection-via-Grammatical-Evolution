import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, input_size, output_size, config):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config['hidden_size'],
            num_layers=config['layers'],
            dropout=config['dropout'],
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(config['hidden_size'] * 2, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])