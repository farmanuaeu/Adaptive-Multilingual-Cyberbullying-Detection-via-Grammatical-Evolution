import pandas as pd
from config import Settings

class DataLoader:
    def __init__(self, dataset_type):
        self.config = Settings.DATASETS[dataset_type]
        
    def load_dataset(self, dataset_type='train'):
        path = getattr(self.config, f'{dataset_type}_path')
        return pd.read_csv(path)
    
    def preprocess_dataset(self, df):
        df['text'] = df['text'].apply(self.clean_text)
        df['language'] = df['text'].apply(self.detect_language)
        return df
    
    def detect_language(self, text):
        try: return detect(text) if len(text) > 10 else 'en'
        except: return 'en'