import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
from bilstm_cnn import BiLSTM_CNN

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset = load_dataset("conll2003")
num_tags = len(set(tag for example in dataset['train'] for tag in example['ner_tags']))
tags = dataset["train"].features["ner_tags"].feature.names
tag2idx = {tag: idx for idx, tag in enumerate(tags)}


#review tokenizers
BATCH_SIZE = 32
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
tokenized_data = tokenizer(dataset["train"]["tokens"]) #disaster

train_tokens = [torch.tensor(example["input_ids"]) for example in tokenized_data]
train_tags = [torch.tensor([tag2idx[tag] for tag in example["ner_tags"]]) for example in dataset["train"]]
train_dataset = list(zip(train_tokens, train_tags))
valid_dataset = tokenized_data["validation"] #fix

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

tag2idx = {'O': 0, 'B-MISC': 1, 'I-MISC': 2, 'B-PER': 3, 'I-PER': 4, 'B-ORG': 5, 'I-ORG': 6, 'B-LOC': 7, 'I-LOC': 8}



EMBEDDING_DIM = 768
HIDDEN_DIM = 256
NUM_CLASSES = 9
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5

model = BiLSTM_CNN(EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=tag2idx['O'])

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    for sentence, tags in train_loader:
        optimizer.zero_grad()
        sentence = sentence.to(device)
        tags = tags.to(device)
        outputs = model(sentence)
        loss = criterion(outputs.view(-1, num_tags), tags.view(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f'Epoch: {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss/len(train_loader)}')


model.eval()
predictions = []
true_labels = []
with torch.no_grad():
    for sentence, tags in valid_loader:
        sentence = sentence.to(device)
        outputs = model(sentence)
        _, predicted = torch.max(outputs, dim=2)
        predicted = predicted.cpu().numpy().tolist()
        tags = tags.cpu().numpy().tolist()
        for i in range(len(tags)):
            predictions.append(predicted[i])
            true_labels.append(tags[i])


#idx2tag = {idx: tag for tag, idx in tag2idx.items()}
#predicted_labels = [[idx2tag[idx] for idx in sent] for sent in predictions]
#true_labels = [[idx2tag[idx] for idx in sent] for sent in true_labels]

