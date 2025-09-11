from src.data_processing.data_annotate import InvoiceDataAutoAnnotator
from src.utils.logger import default_logger as logger
import glob

if __name__ == "__main__":
    csv_files = []

    logger.info("Collecting CSV files from data/batch_1/")

    for file in glob.glob("data/batch_1/*.csv"):
        csv_files.append(file)
        logger.info(f"Found CSV file: {file}")

    for file in glob.glob("data/batch_2/batch_2/*.csv"):
        csv_files.append(file)
        logger.info(f"Found CSV file: {file}")

    logger.info("Initializing InvoiceDataAutoAnnotator...")
    annotator = InvoiceDataAutoAnnotator()

    dataset = annotator.process_csv_files(csv_files)

    logger.info(f"Processed dataset: {dataset}")

    annotator.save_dataset(output_file="data/invoice_ner_dataset.jsonl")

    annotator.analyze_dataset()

    logger.info("Data annotation and saving completed.")