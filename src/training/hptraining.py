from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer
import optuna
from transformers import DataCollatorForTokenClassification
import evaluate
import os
from src.utils.metrics import build_compute_metrics


os.environ["WANDB_DISABLED"] = "true"

class HyperparameterTraining:
    def __init__(self, model_name:str, train_dataset, test_dataset, label_list):
        self.model_name = model_name
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.label_list = label_list
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        self.id2label = {i: label for label, i in self.label2id.items()}

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label_list),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True
        )

        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        self.metric = evaluate.load("seqeval")
        self.compute_metrics = build_compute_metrics(self.id2label, self.metric)
        
    def tokenize_and_align_labels(self, dataset):
        tokenized_inputs = self.tokenizer(dataset['tokens'], truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(dataset['ner_tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)  
                elif word_idx != previous_word_idx:
                    if word_idx < len(label):
                        label_ids.append(self.label2id[label[word_idx]])
                    else:
                        label_ids.append(-100)
                else:
                    if word_idx < len(label):
                        current_label = label[word_idx]
                        label_ids.append(self.label2id[current_label] if current_label.startswith("I-") else -100)
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    
    def hyperparameter_tuning_optuna(self, tokenized_train, tokenized_test, n_trials=20):
        def objective(trial):
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            batch_size = trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32])
            num_epochs = trial.suggest_int("num_train_epochs", 2, 6)
            weight_decay = trial.suggest_float("weight_decay", 0.0, 0.3)
            warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.3)
            
            def model_init():
                return AutoModelForTokenClassification.from_pretrained(
                        self.model_name,
                        num_labels=len(self.label_list),
                        id2label=self.id2label,
                        label2id=self.label2id,
                        ignore_mismatched_sizes=True
                )
            
            training_args = TrainingArguments(
                output_dir=f"./tmp_trial_{trial.number}",
                eval_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=1,  
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=num_epochs,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                logging_dir=None,  
                report_to=None,   
                dataloader_pin_memory=False  
            )
            
            trainer = Trainer(
                model_init=model_init,   
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_test,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                compute_metrics=self.compute_metrics,
            )
            
            trainer.train()
            eval_result = trainer.evaluate()
            
            return eval_result["eval_f1"]
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        print("OPTUNA RESULTS")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best F1 score: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")
        
        return study.best_params, study
