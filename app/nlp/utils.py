import torch
from transformers import AutoModel, AutoTokenizer

def load_hg_model(model_name):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer