import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetFeatureExtractor(nn.Module):
    def __init__(self, point_feature_dim):
        super(PointNetFeatureExtractor, self).__init__()
        self.mlp1 = nn.Conv1d(3, 16, 1)
        self.mlp2 = nn.Conv1d(16, point_feature_dim, 1)
    
    def forward(self, x):
        # x: (B, N, 3)
        B, N, _ = x.size()
        x = x.transpose(1, 2)  # (B, 3, N)
        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))
        return x.transpose(1, 2)  # (B, N, point_feature_dim)

# method 2
class PointCloudSortingRNN(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super(PointCloudSortingRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.feature_extractor = PointNetFeatureExtractor(point_feature_dim=feat_dim)
        self.rnn = nn.GRU(feat_dim, hidden_dim, batch_first=True)
        #self.rnn = nn.LSTM(feat_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # Output one score per point
        
    def forward(self, x):
        # x: (B, N, 3) - Batch, each with N points in 3D
        B, N, _ = x.size()
        
        # Extract features for each point
        features = self.feature_extractor(x)   # (B, N, point_feature_dim)
        
        # Process the points in each frame with RNN
        rnn_out, _ = self.rnn(features)  # (B, N, hidden_dim)
        
        # Generate a score for each point using a fully connected layer
        scores = self.fc(rnn_out.reshape(-1,rnn_out.size(-1))).squeeze(-1).view(B,N)  # (B, N)
        
        # Generate sorting indices based on scores
        #sorting_indices = scores.argsort(dim=1) # (B, N)
        sorting_indices = torch.topk(-scores, k=scores.size(1), dim=1)[1]
        
        return sorting_indices
