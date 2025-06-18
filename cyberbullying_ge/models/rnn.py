import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, output_size, config):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=config['hidden_size'],
            num_layers=config['layers'],
            dropout=config['dropout'],
            batch_first=True
        )
        self.fc = nn.Linear(config['hidden_size'], output_size)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])