class Settings:
    # Dataset configurations
    DATASETS = {
        'english': {
            'train_path': 'data/Fine-grained balanced cyberbullying dataset/train_data.csv',
            'valid_path': 'data/Fine-grained balanced cyberbullying dataset/valid_data.csv',
            'test_path': 'data/Fine-grained balanced cyberbullying dataset/test_data.csv',
            'embedding_model': 'bert-base-uncased'
        },
        'arabic': {
            'train_path': 'data/ArCyC A Fully Annotated Arabic Cyberbullying Corpus/training_data.csv',
            'valid_path': 'data/ArCyC A Fully Annotated Arabic Cyberbullying Corpus/valid_data.csv', 
            'test_path': 'data/ArCyC A Fully Annotated Arabic Cyberbullying Corpus/testing_data.csv',
            'embedding_model': 'xlm-roberta-base'
        }
    }
    
    # Evolutionary parameters
    POPULATION_SIZE = 1000
    GENERATIONS = 100
    MUTATION_RATE = 0.15
    CROSSOVER_RATE = 0.7
    RUNS = 30
    
    # Model constants
    MAX_SEQUENCE_LENGTH = 128
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Fairness constraints
    DEMOGRAPHIC_GROUPS = ['gender', 'age', 'religion']
    MAX_DISPARITY = 0.05