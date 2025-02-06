import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold
import numpy as np

class FoldTrainer:
    """
    Trainer class for handling the training and evaluation process on a single fold of the data.
    
    This class encapsulates the logic for a single fold during cross-validation. It trains the model,
    evaluates it on a validation set at regular intervals, and finally computes performance metrics
    on a held-out test set.
    
    Attributes:
        model (torch.nn.Module): The GAT model to be trained.
        data (Data): The graph data including features, edge indices, and labels.
        optimizer (torch.optim.Optimizer): Optimizer used to update model weights.
        train_mask (Tensor): Boolean mask indicating which nodes are in the training set.
        val_mask (Tensor): Boolean mask indicating which nodes are in the validation set.
        test_mask (Tensor): Boolean mask indicating which nodes are in the test set.
        num_epochs (int): Total number of training epochs.
        eval_interval (int): Interval (in epochs) at which the model is evaluated on the validation set.
    """
    def __init__(self, model, data, optimizer, train_mask, val_mask, test_mask, num_epochs=300, eval_interval=10):
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.num_epochs = num_epochs
        self.eval_interval = eval_interval

    def train_epoch(self):
        """
        Performs one epoch of training.
        
        The method sets the model to training mode, clears previous gradients, computes the loss for the
        training nodes, performs backpropagation, and updates the model parameters.
        
        Returns:
            float: The loss value for the current epoch.
        """
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.data.x, self.data.edge_index)
        loss = F.nll_loss(out[self.train_mask], self.data.y[self.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, mask):
        """
        Evaluates the model on the nodes specified by the given mask.
        
        The method computes performance metrics including accuracy, precision, recall, F1-score,
        and ROC AUC score.
        
        Args:
            mask (Tensor): Boolean mask indicating the subset of nodes for evaluation.
            
        Returns:
            tuple: A tuple containing accuracy, precision, recall, F1-score, and ROC AUC score.
        """
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            out = F.softmax(out, dim=1)
            pred = out[mask].max(1)[1]
            true = self.data.y[mask]
            
            accuracy = accuracy_score(true.cpu(), pred.cpu())
            precision = precision_score(true.cpu(), pred.cpu(), average='weighted', zero_division=0)
            recall = recall_score(true.cpu(), pred.cpu(), average='weighted', zero_division=0)
            f1 = f1_score(true.cpu(), pred.cpu(), average='weighted', zero_division=0)
            auc = roc_auc_score(true.cpu(), out[mask].cpu(), multi_class='ovr', average='weighted')
        
        return accuracy, precision, recall, f1, auc

    def run(self):
        """
        Executes the training loop and performs periodic evaluation.
        
        The model is trained for the specified number of epochs. After each eval_interval, the model is
        evaluated on the validation set, and metrics are printed. Once training is complete, the model
        is evaluated on the test set and test metrics are returned.
        
        Returns:
            tuple: Performance metrics (accuracy, precision, recall, F1-score, ROC AUC) computed on the test set.
        """
        for epoch in range(1, self.num_epochs + 1):
            loss = self.train_epoch()
            if epoch % self.eval_interval == 0:
                val_metrics = self.evaluate(self.val_mask)
                print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')
                print(f'Validation Metrics -- Acc: {val_metrics[0]:.4f}, Precision: {val_metrics[1]:.4f}, '
                      f'Recall: {val_metrics[2]:.4f}, F1: {val_metrics[3]:.4f}, AUC: {val_metrics[4]:.4f}')
        
        test_metrics = self.evaluate(self.test_mask)
        print(f'\nTest Metrics -- Acc: {test_metrics[0]:.4f}, Precision: {test_metrics[1]:.4f}, '
              f'Recall: {test_metrics[2]:.4f}, F1: {test_metrics[3]:.4f}, AUC: {test_metrics[4]:.4f}\n')
        return test_metrics

class GraphTrainer:
    """
    Manages dataset loading, K-Fold cross-validation, and model training across folds.
    
    This class is responsible for:
      - Loading the specified Planetoid dataset.
      - Splitting the nodes into training, validation, and test sets using K-Fold cross-validation.
      - Initializing and training a new model instance for each fold.
      - Aggregating and reporting average performance metrics across all folds.
    
    Attributes:
        dataset_name (str): Name of the Planetoid dataset (e.g., 'Cora', 'Citeseer', 'PubMed').
        dataset_root (str): Path to the root directory where the dataset is stored.
        model_class (class): The model class to be instantiated (e.g., GAT).
        hidden_channels (int): Number of hidden channels in the GATv2 layers.
        num_heads (int): Number of attention heads in the first GATv2 layer.
        feature_dropout (float): Dropout rate for node features.
        attention_dropout (float): Dropout rate used in the attention mechanism.
        num_epochs (int): Total number of training epochs for each fold.
        k_folds (int): Number of folds for K-Fold cross-validation.
        eval_interval (int): Frequency (in epochs) at which the model is evaluated on the validation set.
        learning_rate (float): Learning rate for the optimizer.
        dataset (Planetoid): Loaded dataset object.
        data (Data): The graph data extracted from the dataset.
    """
    def __init__(self, dataset_name, dataset_root, model_class, 
                 hidden_channels=16, num_heads=8, feature_dropout=0.6, 
                 attention_dropout=0.6, num_epochs=1000, k_folds=5, 
                 eval_interval=25, learning_rate=0.001):
        self.dataset_name = dataset_name
        self.dataset_root = dataset_root
        self.model_class = model_class
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.feature_dropout = feature_dropout
        self.attention_dropout = attention_dropout
        self.num_epochs = num_epochs
        self.k_folds = k_folds
        self.eval_interval = eval_interval
        self.learning_rate = learning_rate

        # Load the Planetoid dataset.
        self.dataset = Planetoid(root=self.dataset_root, name=self.dataset_name)
        self.data = self.dataset[0]
        # Ensure a test mask exists; if not, create one that uses 20% of the nodes.
        if not hasattr(self.data, 'test_mask'):
            self.data.test_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
            self.data.test_mask[:int(0.2 * self.data.num_nodes)] = True

    def cross_validate(self):
        """
        Performs K-Fold cross-validation on the graph data.
        
        The method splits the node features into k folds. For each fold:
          - It creates training and validation masks.
          - It initializes a new model and optimizer.
          - It trains the model on the training set and evaluates it on the validation set.
          - It computes performance metrics on the test set.
        
        Returns:
            dict: A dictionary containing the averaged test metrics (accuracy, precision, recall, F1-score, AUC)
                  across all folds.
        """
        kfold = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.data.x)):
            print(f"\nTraining fold {fold + 1}/{self.k_folds}")

            # Create boolean masks for training and validation nodes.
            train_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
            train_mask[train_idx] = True
            val_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
            val_mask[val_idx] = True
            test_mask = self.data.test_mask  # Use the predefined test mask.

            # Instantiate a new model and optimizer for this fold.
            model = self.model_class(
                in_channels=self.data.num_features,
                hidden_channels=self.hidden_channels,
                out_channels=self.dataset.num_classes,
                heads=self.num_heads,
                dropout=self.feature_dropout,
                attention_dropout=self.attention_dropout
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
            
            # Create a FoldTrainer to handle training and evaluation for this fold.
            fold_trainer = FoldTrainer(model, self.data, optimizer, train_mask, val_mask, test_mask,
                                       num_epochs=self.num_epochs, eval_interval=self.eval_interval)
            test_metrics = fold_trainer.run()
            
            # Collect test metrics for this fold.
            fold_results.append({
                'test_acc': test_metrics[0],
                'test_precision': test_metrics[1],
                'test_recall': test_metrics[2],
                'test_f1': test_metrics[3],
                'test_auc': test_metrics[4]
            })
        
        # Compute the average metrics across all folds.
        avg_results = {key: np.mean([fold[key] for fold in fold_results])
                       for key in fold_results[0].keys()}
        return avg_results