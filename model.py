import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class GAT(torch.nn.Module):
    """
    Graph Attention Network (GATv2) implementation.

    This model consists of two graph attention layers. The first layer uses multiple
    attention heads, whose outputs are concatenated and then normalized using batch normalization.
    The second layer aggregates these features (without concatenation) and applies another batch
    normalization before producing log-softmax outputs for classification.
    
    Attributes:
        dropout (float): Dropout rate applied to intermediate features.
        conv1 (GATv2Conv): First graph attention layer.
        conv2 (GATv2Conv): Second graph attention layer.
        batch_norm1 (BatchNorm1d): Batch normalization applied after the first layer.
        batch_norm2 (BatchNorm1d): Batch normalization applied after the second layer.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.3, attention_dropout=0.3):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=attention_dropout)
        self.conv2 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=attention_dropout)
        self.batch_norm1 = nn.BatchNorm1d(hidden_channels * heads)
        self.batch_norm2 = nn.BatchNorm1d(out_channels)

    def forward(self, x, edge_index):
        """
        Forward pass of the GAT model.
        
        Args:
            x (Tensor): Node feature matrix.
            edge_index (Tensor): Graph connectivity (edge indices).

        Returns:
            Tensor: Log-probabilities for each node belonging to each class.
        """
        # First graph attention layer with batch normalization and activation
        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second graph attention layer with batch normalization
        x = self.conv2(x, edge_index)
        x = self.batch_norm2(x)
        return F.log_softmax(x, dim=1)