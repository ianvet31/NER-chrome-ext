import torch
import torch.nn as nn
from transformers import BertModel

class BiLSTM_CNN(nn.Module): #+bert 
    def __init__(self, embedding_dim, hidden_dim, num_classes):
        super(BiLSTM_CNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.bilstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.conv1d = nn.Conv1d(in_channels=2*hidden_dim, out_channels=2*hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_features=2*hidden_dim, out_features=num_classes)
    
    def forward(self, inputs):
        encoded_layers, _ = self.bert(inputs['input_ids'], attention_mask=inputs['attention_mask'], return_dict=False)
        bert_output = encoded_layers[-1] 
        lstm_output, _ = self.bilstm(bert_output)
        conv_output = self.conv1d(lstm_output.transpose(1, 2))
        conv_output = self.relu(conv_output)
        pool_output = torch.max(conv_output, dim=2)[0]
        pool_output = self.dropout(pool_output)
        logits = self.fc(pool_output)
        return logits