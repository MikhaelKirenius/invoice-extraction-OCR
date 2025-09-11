from datasets import load_dataset
from src.utils.logger import default_logger as logger

class DatasetLoader:
    def __init__(self, file_path: str):
        self.dataset = None
        self.file_path = file_path

    def load_and_split_data(self):
        dataset = load_dataset("json", data_files=self.file_path)

        tmp = dataset.train_test_split(test_size=0.2, seed=42)
        dataset_train = tmp['train'].train_test_split(test_size=0.2, seed=42)

        self.train_dataset = dataset_train['train']
        self.val_dataset = dataset_train['validation']
        self.test_dataset = tmp['test']

        return self.train_dataset, self.val_dataset, self.test_dataset

    def get_label_list(self):
        try:
            logger.info("Extracting Label List From Dataset")
            unique_labels = set(label for row in self.train_dataset['ner_tags'] for label in row)
            label_list = sorted(list(unique_labels))
            return label_list
        except:
            logger.error('Load Data First')