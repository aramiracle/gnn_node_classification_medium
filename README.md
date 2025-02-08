# Graph Attention Network (GAT) with K-Fold Cross-Validation

This repository contains an implementation of a Graph Attention Network (GAT) with K-Fold cross-validation using the Planetoid datasets. The GAT model is used for graph-based node classification, and the code supports evaluating the model on multiple datasets using cross-validation.

## Table of Contents
- [Requirements](#requirements)
- [Usage](#usage)
- [Files and Functions](#files-and-functions)
- [Model Description](#model-description)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Medium](#medium)

## Requirements

To run this project, you need to have the following dependencies installed:

- Python 3.7+
- PyTorch
- PyTorch Geometric
- Scikit-learn
- NumPy

You can install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage

### Training and Evaluation on Multiple Datasets

1. Clone the repository to your local machine:

   ```bash
   git clone git@github.com:aramiracle/gnn_node_classification_medium.git
   cd gnn_node_classification_medium
   ```

2. Modify the dataset_root path in main.py to point to the location of your Planetoid dataset (e.g., data/Planetoid).

3. Run the main.py script to evaluate the model on multiple Planetoid datasets (Cora, Citeseer, and PubMed) with K-Fold cross-validation.

```bash
python main.py
```

The script will train the GAT model, perform cross-validation on each dataset, and print the averaged evaluation metrics.

## Files and Functions

### `trainer.py`
This file contains the `FoldTrainer` and `GraphTrainer` classes for managing the training and evaluation process.

- **`FoldTrainer`**: Handles training, evaluation, and performance metric computation for a single fold of the data.
  - `train_epoch()`: Trains the model for one epoch.
  - `evaluate()`: Evaluates the model on a given set of nodes.
  - `run()`: Runs the training loop, periodically evaluates the model, and returns test set metrics.

- **`GraphTrainer`**: Manages the dataset loading, K-Fold cross-validation, and model training across folds.
  - `cross_validate()`: Splits the dataset into K folds, trains a new model for each fold, and computes the average performance metrics.

### `model.py`
This file contains the `GAT` class for implementing the Graph Attention Network (GATv2).

- **`GAT`**: A GATv2 model with two graph attention layers. The first layer uses multiple attention heads, and the second aggregates the features for classification.
  - `forward()`: Defines the forward pass of the model, including attention layers and batch normalization.

### `main.py`
The main script to initialize the `GraphTrainer`, perform K-Fold cross-validation, and evaluate the GAT model on multiple Planetoid datasets.

## Model Description

The model used in this repository is based on the Graph Attention Network (GATv2). It includes two graph attention layers:

- **First Layer**: Multiple attention heads are used, and their outputs are concatenated, followed by batch normalization.
- **Second Layer**: Aggregates features without concatenation, applies batch normalization, and produces log-softmax outputs for classification.

## Evaluation Metrics

During training and evaluation, the following metrics are computed:

- **Accuracy**: The proportion of correct predictions.
- **Precision**: The weighted average of precision across all classes.
- **Recall**: The weighted average of recall across all classes.
- **F1-score**: The weighted average of F1 scores across all classes.
- **AUC (Area Under the ROC Curve)**: A measure of the model's ability to distinguish between classes.

These metrics are calculated for each fold during cross-validation and averaged across all folds.

## Results

After running the script, the final test results for all datasets will be displayed, including the average accuracy, precision, recall, F1-score, and AUC across all folds.

Example output:

```bash
Final Test Results for All Datasets:

Cora:
test_acc: 0.9550
test_precision: 0.9551
test_recall: 0.9550
test_f1: 0.9550
test_auc: 0.9966

Citeseer:
test_acc: 0.9116
test_precision: 0.9116
test_recall: 0.9116
test_f1: 0.9108
test_auc: 0.9897

PubMed:
test_acc: 0.8950
test_precision: 0.8956
test_recall: 0.8950
test_f1: 0.8948
test_auc: 0.9777
```

## Medium

I also write a comprehensive description about this project on medium website. Follow the [link](https://medium.com/@a.r.amouzad.m/how-to-get-state-of-the-art-result-on-node-classification-with-graph-neural-networks-c74cb373cb66)