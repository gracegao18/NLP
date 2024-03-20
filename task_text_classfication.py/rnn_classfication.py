from unicodedata import bidirectional
from torch import nn

class TextClassficationModel(nn.Module):
    def __init__(self, 
        vocab_size, 
        emb_hidden_size, 
        rnn_hidden_size, 
        num_layers, 
        num_class):
        super(TextClassficationModel, self).__init__()
        # Embedding模型
        self.embedding = nn.Embedding(
            vocab_size, 
            emb_hidden_size, 
            padding_idx = 0)
        # LSTM模型
        self.rnn = nn.LSTM(         
            input_size = emb_hidden_size,      
            hidden_size = rnn_hidden_size,     
            num_layers = num_layers,       
            batch_first = True, 
            # 双向结构
            bidirectional = True 
        )
        # 双向结构 -> rnn_hidden_size * 2
        self.out = nn.Linear(rnn_hidden_size * 2, num_class) 

    def forward(self, x):
        out = self.embedding(x)
        r_out, _ = self.rnn(out)
        out = self.out(r_out[:,-1,:])
        return out