from transformers import AutoTokenizer, AutoModel
from config import Settings

class EmbeddingGenerator:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(Settings.DEVICE)
        self.model.eval()
        
    def generate_embeddings(self, texts):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=Settings.MAX_SEQUENCE_LENGTH,
            return_tensors="pt"
        ).to(Settings.DEVICE)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()