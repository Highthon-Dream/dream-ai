import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sentencepiece as spm

def load_labels():
    with open("./labels.txt", "r", encoding="utf-8") as f:
        labels = f.read().split("\n")
    return labels

def load_tokenizer(vocab_file="wiki_tokenizer.model"):
    sp = spm.SentencePieceProcessor()
    sp.load(vocab_file)

    return sp

tokenizer = load_tokenizer()

def tokenize(tokenizer, line):
    return tokenizer.encode_as_ids(line)
    
class EmbeddingMLP(nn.Module):
    def __init__(self, feature_in, embd_size, label_size, window_size):
        super().__init__()

        self.emb_layer = nn.Embedding(feature_in, embd_size)

        self.mlp_layer = nn.Sequential(
            nn.Linear(embd_size, 64),
            nn.ReLU(),
            nn.Linear(64, label_size)
        )

    def forward(self, x):
        x = self.emb_layer(x)
        x = x.sum(1)
        x = self.mlp_layer(x)

        return x

    def save(self):
        torch.save(self.emb_layer.state_dict(), "pretrained-embedding.pt")

window_size = 20

class CustomDataset():
    def __init__(self):
        with open("./datas.txt", "r", encoding="utf-8") as f:
            content = f.read()

        with open("./ys.txt", "r", encoding="utf-8") as f:
            contett = f.read()

        self.x = []
        self.y = []

        for (goal, label) in zip(content.split("\n"), contett.split("\n")):
            # _, goal, labels = line.split("|")
            goal = f"\"{' '.join(goal.split('	')[1:]).strip()}\""
            self.x.append(goal)
            # labels = [int(label) for label in labels.strip().split(" ")]
            self.y.append(int(label.split(" ")[-1]))
        
    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]

        tokenized = torch.tensor(tokenize(tokenizer, x))
        if tokenized.shape[0] > window_size:
            tokenized = tokenized[:window_size]
        x = F.pad(tokenized, (0, window_size-tokenized.shape[0]), "constant")
        y = torch.tensor(y)

        return x, y
        
    def __len__(self): return len(self.x)

dataset = CustomDataset()
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
# dataloader = DataLoader(dataset, 64, shuffle=True)
trainloader = DataLoader(train_set, 64, shuffle=True)
testloader = DataLoader(test_set, 64, shuffle=False)

model = EmbeddingMLP(10000, 128, 5, window_size=window_size)
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(40):
    for (x, y) in trainloader:
        pred = model(x)
        loss = nn.functional.cross_entropy(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"epoch: {epoch} | {loss.item()}")
    
    for (x, y) in testloader:
        pred = model(x)
        print(((pred.softmax(1).argmax(1)==y)/y.shape[0]*100).sum())

    model.save()
