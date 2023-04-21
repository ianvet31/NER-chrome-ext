import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import datasets
from tqdm import tqdm

# Load the CoNLL 2003 dataset
conll2003 = datasets.load_dataset("conll2003")

# Define the word-level feature extraction function
def extract_features(example):
    words = example["tokens"]
    features = []
    for i, word in enumerate(words):
        feature = {
            "word": word,
            "is_first": i == 0,
            "is_last": i == len(words) - 1,
            "is_capitalized": word[0].upper() == word[0],
            "is_all_caps": word.upper() == word,
            "is_all_lower": word.lower() == word,
            "prefix-1": word[0],
            "prefix-2": word[:2],
            "prefix-3": word[:3],
            "suffix-1": word[-1],
            "suffix-2": word[-2:],
            "suffix-3": word[-3:],
            "prev_word": words[i-1] if i > 0 else "<START>",
            "next_word": words[i+1] if i < len(words)-1 else "<END>",
        }
        features.append(feature)
    return features, example["ner_tags"]

# Apply the feature extraction function to the dataset
train_data = conll2003["train"].map(extract_features)
val_data = conll2003["validation"].map(extract_features)
test_data = conll2003["test"].map(extract_features)

# Define the NER model
class NERModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        x = self.fc(lstm_out)
        return x

# Define the training and evaluation functions
def train(model, data_loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for batch in tqdm(data_loader, desc="Training", unit="batch"):
        inputs = torch.tensor(batch[0], dtype=torch.long).to(device)
        targets = torch.tensor(batch[1], dtype=torch.long).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, outputs.shape[-1]), targets.view(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

def evaluate(model, data_loader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", unit="batch"):
            inputs = torch.tensor(batch[0], dtype=torch.long).to(device)
            targets = torch.tensor(batch[1], dtype=torch.long).to(device)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), targets.view(-1))
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

# Set up the device and hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

input_dim = len(conll2003["train"][0][0])
hidden_dim = 128
output_dim = len(conll2003["train"].features["ner_tags"].feature.names)

lr = 0.001
batch_size = 64
num_epochs = 10



train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)



model = NERModel(input_dim, hidden_dim, output_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion)
    val_loss = evaluate(model, val_loader, criterion)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    test_loss = evaluate(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}")