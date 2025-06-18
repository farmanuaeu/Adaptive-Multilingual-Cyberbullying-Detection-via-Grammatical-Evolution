GRAMMAR = {
    'model_type': ['mlp', 'rnn', 'lstm', 'bilstm'],
    'layers': [2, 3, 4],
    'hidden_size': [128, 256, 512],
    'activation': ['relu', 'gelu', 'leaky_relu'],
    'dropout': [0.2, 0.3, 0.4],
    'optimizer': ['adam', 'adamw', 'sgd']
}

DIALECT_MAPPINGS = {
    'arabic': {
        'شو': 'ما',
        'حلوة أوي': 'جميلة جدا',
        # Add more dialect mappings
    },
    'english': {}
}