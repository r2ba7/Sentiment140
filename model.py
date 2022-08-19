from imports import *
from utils import Config

class Classifier(nn.Module):
    def __init__(self, input_size):
        """
        Initializes the classifier's parameters..
        """
        super().__init__()
        self.input_size = input_size #vocab_size
        self.hidden_dim = Config.HIDDEN_SIZE
        self.output_size = Config.OUTPUT_SIZE
        self.LSTM_layers = Config.LSTM_LAYERS
        self.dropout = nn.Dropout(0.5)
        self.embedding = nn.Embedding(self.input_size, self.hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.LSTM_layers, batch_first=True)
        self.fc1 = nn.Linear(in_features=self.hidden_dim, out_features=64)
        self.fc2 = nn.Linear(64, self.output_size)

    def forward(self, x):
        """
        Forward pass.
        """
        h0 = torch.zeros(self.LSTM_layers, x.size(0), self.hidden_dim, device=x.device).float()
        c0 = torch.zeros(self.LSTM_layers, x.size(0), self.hidden_dim, device=x.device).float()
        # h0 = torch.zeros(self.LSTM_layers, x.size(0), self.hidden_dim).float()
        # c0 = torch.zeros(self.LSTM_layers, x.size(0), self.hidden_dim).float()
        torch.nn.init.xavier_normal_(h0)
        torch.nn.init.xavier_normal_(c0)
        out = self.embedding(x)
        out, _ = self.lstm(out, (h0,c0))
        out = self.dropout(out)
        out = torch.relu_(self.fc1(out[:,-1,:]))
        out = self.dropout(out)
        out = torch.sigmoid(self.fc2(out))       
        return out
