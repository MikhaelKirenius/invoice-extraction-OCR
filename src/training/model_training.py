from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForTokenClassification
import evaluate
from src.utils.metrics import build_compute_metrics

import os 
os.environ["WANDB_DISABLED"] = "true"

class FinalModelTrainer:
    def __init__(self, model_name, label_list):
        self.model_name = model_name

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

    def train_with_best_params(self,best_params, tokenized_train, tokenized_test,output_dir="./final_model" ):
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            learning_rate=best_params["learning_rate"],
            per_device_train_batch_size=best_params["per_device_train_batch_size"],
            num_train_epochs=best_params["num_train_epochs"],
            weight_decay=best_params["weight_decay"],
            warmup_ratio=best_params["warmup_ratio"],
            load_best_model_at_end=True,
            metric_for_best_model="f1",
        )

        trainer = Trainer(
            model_init=lambda: AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.label_list),
                id2label=self.id2label,
                label2id=self.label2id,
                ignore_mismatched_sizes=True
            ),
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics
        )

        trainer.train()
        
        return trainer
