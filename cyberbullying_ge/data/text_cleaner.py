import re
from config import grammar

class TextCleaner:
    def __init__(self, language):
        self.dialect_map = grammar.DIALECT_MAPPINGS[language]
        
    def clean_text(self, text):
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^\w\s@#]', '', text)
        return self.normalize_dialect(text.strip().lower())
    
    def normalize_dialect(self, text):
        for dialect, standard in self.dialect_map.items():
            text = text.replace(dialect, standard)
        return text