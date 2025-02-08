from trainer import GraphTrainer
from model import GAT

def main():
    """
    Main function to run the training and evaluation process on multiple datasets.
    
    This function iterates over a list of specified datasets. For each dataset, it initializes a
    GraphTrainer, performs K-Fold cross-validation, and prints the average test metrics.
    """
    # List of Planetoid datasets to evaluate.
    datasets = ['Cora', 'Citeseer', 'PubMed']
    dataset_root = 'data/Planetoid'

    # Hyperparameters for the GAT model and training process.
    hidden_channels = 16
    num_heads = 8
    feature_dropout = 0.6
    attention_dropout = 0.6
    num_epochs = 1000
    k_folds = 5
    eval_interval = 25
    learning_rate = 0.001
    random_seed = 42  # Random seed for reproducibility

    final_results = {}

    # Iterate through each dataset and perform cross-validation.
    for dataset_name in datasets:
        print(f"\nEvaluating on {dataset_name} dataset")
        
        # Initialize the GraphTrainer with the specified hyperparameters.
        trainer = GraphTrainer(
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            model_class=GAT,
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            feature_dropout=feature_dropout,
            attention_dropout=attention_dropout,
            num_epochs=num_epochs,
            k_folds=k_folds,
            eval_interval=eval_interval,
            learning_rate=learning_rate,
            random_seed=random_seed  # Ensure reproducibility for each dataset
        )
        
        # Perform cross-validation and get the averaged results.
        avg_results = trainer.cross_validate()
        final_results[dataset_name] = avg_results

    # Display the averaged test results for all datasets.
    print("\nFinal Test Results for All Datasets:")
    for dataset_name, results in final_results.items():
        print(f"\n{dataset_name}:")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")

if __name__ == '__main__':
    main()
