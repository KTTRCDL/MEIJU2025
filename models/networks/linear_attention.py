import torch
import torch.nn as nn

class LinearAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(LinearAttention, self).__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, features):
        attn_scores = self.attention_weights(features)
        attn_weights = torch.softmax(attn_scores, dim=1)
        weighted_features = features * attn_weights
        output = weighted_features.sum(dim=1)
        return output, attn_weights