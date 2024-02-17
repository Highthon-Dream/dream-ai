import numpy as np
import os

class VectorDB():
    def __init__(self, save_path="./vectordb.npy"):
        self.save_path = save_path
        self.db = np.array([], dtype=np.float32)
        if os.path.exists(save_path):
            self.db = np.load(save_path)

    def add(self, emb):
        self.emb = np.append(self.emb, emb, axis=0)

    def search_similar(self, emb):
        pass
    
    def delete(self, idx):
        assert len(self) < idx
        self.emb = np.delete(self.emb, idx, axis=1)

    def update(self, idx, emb):
        self.db[idx] = emb
    
    def save(self):
        return np.save(self.save_path)

    def __len__(self): return self.emb.shape[0]