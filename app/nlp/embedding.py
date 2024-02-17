import torch
from transformers import AutoModel, AutoTokenizer
from functools import lru_cache

from .utils import load_hg_model

class EmbeddingModel():
    def __init__(self):
        self.model, self.tokenizer = load_hg_model('BM-K/KoSimCSE-roberta')

    def get_embs(self, sentences: list[str]):
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        embeddings, _ = self.model(**inputs, return_dict=False)
        embeddings = embeddings.view(embeddings.shape[0], -1)
        return embeddings

    def __call__(self, sentences): return self.get_embs(sentences)

    def calc_sim(self, a, b):
        a_norm = a / a.norm(dim=1)[:, None]
        b_norm = b / b.norm(dim=1)[:, None]
        return torch.mm(a_norm, b_norm.transpose(0, 1)) * 100

@lru_cache
def load_embedding_model():
    return EmbeddingModel()

if __name__ == "__main__":
    emb_model = EmbeddingModel()
    # 3, 768
    print(emb_model(["안녕", "반가워"]).shape)