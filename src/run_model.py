import os
import argparse
import numpy as np
from datasets import DatasetDict

from src.training.load_dataset import DatasetLoader
from src.training.hptraining import HyperparameterTraining
from src.training.model_training import FinalModelTrainer
from src.utils.metrics import build_compute_metrics
from src.utils.logger import default_logger as logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, default='./data/invoice_ner_dataset.jsonl')
    parser.add_argument('--model_name', type=str, default='bert-base-cased')
    parser.add_argument('--n_trials', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='./final_model')
    parser.add_argument('--final_model_dir', type=str, default='models/final_model_NER_best')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.environ['WANDB_DISABLED'] = 'true'

    logger.info("Loading and Splitting Dataset")
    dataset_loader = DatasetLoader(file_path=args.data_path)
    train_dataset, val_dataset, test_dataset = dataset_loader.load_and_split_data()
    label_list = dataset_loader.get_label_list()

    logger.info("Initializing Hyperparameter Training")
    hptraining = HyperparameterTraining(
        model_name=args.model_name,
        train_dataset=train_dataset,
        test_dataset=val_dataset,
        label_list=label_list
    )

    logger.info("Tokenizing and Aligning Labels for Training Set")
    tokenized_train = hptraining.tokenize_and_align_labels(hptraining.train_dataset)

    logger.info("Tokenizing and Aligning Labels for Validation Set")
    tokenized_val = hptraining.tokenize_and_align_labels(hptraining.val_dataset)

    logger.info("Tokenizing and Aligning Labels for Test Set")
    tokenized_test = hptraining.tokenize_and_align_labels(test_dataset)

    logger.info("Starting Hyperparameter Search")
    best_params, study = hptraining.hyperparameter_tuning_optuna(
        tokenized_train=tokenized_train,
        tokenized_test=tokenized_val,
        n_trials=args.n_trials
    )

    logger.info("Initialized Final Model")
    final_model_trainer = FinalModelTrainer(
        model_name=args.model_name,
        label_list=label_list
    )

    logger.info("Training Final Model with Best Hyperparameters")
    trainer = final_model_trainer.train_with_best_params(
        best_params=best_params,
        tokenized_train=tokenized_train,
        tokenized_test=tokenized_test,
        output_dir=args.output_dir
    )

    logger.info("Process Completed Successfully")

    logger.info("Evaluating Final Model on Test Set")

    eval_results = trainer.evaluate()

    print("Final Model Evaluation Results:")
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    logger.info("Saving Final Model")

    trainer.save_model(args.final_model_dir)
    trainer.tokenizer.save_pretrained(args.final_model_dir)

if __name__ == "__main__":
    main()