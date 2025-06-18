import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, output_size, config):
        super().__init__()
        layers = []
        current_size = input_size
        
        for _ in range(config['layers']):
            layers += [
                nn.Linear(current_size, config['hidden_size']),
                self._get_activation(config['activation']),
                nn.Dropout(config['dropout'])
            ]
            current_size = config['hidden_size']
            
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(current_size, output_size)
        
    def _get_activation(self, name):
        return {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(0.2)
        }[name]
        
    def forward(self, x):
        return self.fc(self.layers(x))