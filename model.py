import torch
import torch.nn as nn
from transformer import *
#from Language_Diarization_main.new_codes.transformer import *
class X_Base_E2E_LD(nn.Module):
    def __init__(self, feat_dim,
                 d_k, d_v, d_ff, n_heads=4,
                 dropout=0.1,n_lang=3, max_seq_len=140,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
      
        super(X_Base_E2E_LD, self).__init__()
        self.feat_dim = feat_dim
        self.device = device
        
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(1536, 3000) 
        self.bn1 = nn.BatchNorm1d(3000, momentum=0.1, affine=False)
        self.fc2 = nn.Linear(3000, feat_dim)
        self.bn2 = nn.BatchNorm1d(feat_dim, momentum=0.1, affine=False)
        self.fc3 = nn.Linear(feat_dim, feat_dim)
        self.bn3 = nn.BatchNorm1d(feat_dim, momentum=0.1, affine=False)  # momentum=0.5 in asv-subtools
        self.fc4 = nn.Linear(feat_dim, n_lang)

        # attention module
        self.layernorm1 = LayerNorm(feat_dim)
        self.pos_encoding = PositionalEncoding(max_seq_len=max_seq_len, features_dim=256, device=device)
        #self.pos_encoding = PositionalEncoding(max_seq_len=max_seq_len, features_dim=256)
        self.layernorm2 = LayerNorm(feat_dim)
        self.attention_block1 = EncoderBlock(feat_dim, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.attention_block2 = EncoderBlock(feat_dim, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.attention_block3 = EncoderBlock(feat_dim, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.attention_block4 = EncoderBlock(feat_dim, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.output_fc = nn.Linear(feat_dim, n_lang)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, seq_len, atten_mask, eps=1e-5):

        b, s, f = x.size()
        x = x.view(b, int(s/10), 10, f)
        xm = torch.mean(x,2)         
        xs = torch.std(x,2)
        x = torch.cat((xm, xs),2)
        x = x.view(-1, 1536)

        x = self.bn1(F.relu(self.fc1(x)))
        embedding = self.fc2(x)
        x = self.bn2(F.relu(embedding))
        x = self.dropout(x)
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.dropout(x)
        cnn_output = self.fc4(x)

        embedding = embedding.view(b, int(s/10), self.feat_dim)
        output = self.layernorm1(embedding)
        output = self.pos_encoding(output,seq_len)
        output = self.layernorm2(output)
        output, _ = self.attention_block1(output, atten_mask)
        output, _ = self.attention_block2(output, atten_mask)
        output, _ = self.attention_block3(output, atten_mask)
        output, _ = self.attention_block4(output, atten_mask)
        output = self.sigmoid(self.output_fc(output))

        return output, cnn_output

class X_Attention_E2E_LD(nn.Module):
    def __init__(self, feat_dim,
                 d_k, d_v, d_ff, n_heads=4,
                 dropout=0.1,n_lang=3, max_seq_len=140,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
      
        super(X_Attention_E2E_LD, self).__init__()
        self.feat_dim = feat_dim
        self.device = device

        self.linear1 = nn.Linear(99, 99)
        self.linear2 = nn.Linear(99, 99)
        self.linear3 = nn.Linear(99, 1) # Attention network starts here.
        self.linear4 = nn.Linear(99, 1536)

        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(1536, 3000) 
        self.bn1 = nn.BatchNorm1d(3000, momentum=0.1, affine=False)
        self.fc2 = nn.Linear(3000, feat_dim)
        self.bn2 = nn.BatchNorm1d(feat_dim, momentum=0.1, affine=False)
        self.fc3 = nn.Linear(feat_dim, feat_dim)
        self.bn3 = nn.BatchNorm1d(feat_dim, momentum=0.1, affine=False)
        self.fc4 = nn.Linear(feat_dim, n_lang)

        # attention module
        self.layernorm1 = LayerNorm(feat_dim)
        self.pos_encoding = PositionalEncoding(max_seq_len=max_seq_len, features_dim=256, device=device)
        self.layernorm2 = LayerNorm(feat_dim)
        self.attention_block1 = EncoderBlock(feat_dim, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.attention_block2 = EncoderBlock(feat_dim, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.attention_block3 = EncoderBlock(feat_dim, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.attention_block4 = EncoderBlock(feat_dim, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.output_fc = nn.Linear(feat_dim, n_lang)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, seq_len, atten_mask, eps=1e-5):
        b, s, f = x.size()
        x = x.view(-1, 99)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        # Storing the results
        # These are the H vectors....bottle neck features. 
        x_orig = x

        # Now calcualte the weights for each frame.
        x = F.relu(self.linear3(x)) # ([32960, 1])

        # Multiply each frame with its corressponding weight and add frames belonging to its respective utterance!
        start = 0
        new_x = torch.zeros(int(x_orig.shape[0]/10), x_orig.shape[1]).to(device=self.device)
        for i in range(int(x_orig.shape[0]/10)):
          sub_i = x_orig[start:start+10, :]
          sub_wts = F.softmax(x[start:start+10, :],dim=0)
          sub_i = sub_i * sub_wts
          new_x[i,:] = torch.sum(sub_i, 0)
          start = start + 10
        x = F.relu(self.linear4(new_x))
        x = self.bn1(F.relu(self.fc1(x)))
        embedding = self.fc2(x)
        x = self.bn2(F.relu(embedding))
        x = self.dropout(x)
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.dropout(x)
        cnn_output = self.fc4(x)

        embedding = embedding.view(b, int(s/10), self.feat_dim)
        output = self.layernorm1(embedding)
        output = self.pos_encoding(output,seq_len)
        output = self.layernorm2(output)
        output, _ = self.attention_block1(output, atten_mask)
        output, _ = self.attention_block2(output, atten_mask)
        output, _ = self.attention_block3(output, atten_mask)
        output, _ = self.attention_block4(output, atten_mask)
        output = self.sigmoid(self.output_fc(output))
        return output, cnn_output