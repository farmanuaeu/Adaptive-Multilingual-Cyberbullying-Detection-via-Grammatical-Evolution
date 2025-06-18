from .mlp import MLP
from .rnn import RNN
from .lstm import LSTM
from .bilstm import BiLSTM

class ModelFactory:
    @staticmethod
    def create_model(config, input_size, output_size):
        model_type = config['model_type']
        
        return {
            'mlp': MLP,
            'rnn': RNN,
            'lstm': LSTM,
            'bilstm': BiLSTM
        }[model_type](input_size, output_size, config)