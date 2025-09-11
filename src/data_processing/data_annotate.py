import pandas as pd
import json
import re
from typing import List, Dict, Optional
from collections import Counter

class InvoiceDataAutoAnnotator:
    def __init__(self, case_sensitive: bool = False):
        self.case_sensitive = case_sensitive
        self.dataset = []
        self.error_count = 0
        self.id_counter = 0

    @staticmethod
    def normalize_number_format(value: str) -> List[str]:
        if not value:
            return []
        formats = [value]
        
        if '.' in value and not ',' in value:
            comma_version = value.replace('.',',')
            formats.append(comma_version)

        if ',' in value and not '.' in value:
            dot_version = value.replace(',','.')
            formats.append(dot_version)

        return formats
    
    def find_entity_in_tokens(self, tokens: List[str], entity_value: str) -> List[int]:
        if not entity_value:
            return []
        
        positions = []
        possible_values = self.normalize_number_format(entity_value)
        for value in possible_values:
            value_tokens = value.split()
            for i in range(len(tokens) -  len(value_tokens) + 1):
                match = True
                for j, value_token in enumerate(value_tokens):
                    token = tokens[i + j]
                    if self.case_sensitive:
                        if token != value_token:
                            match = False
                            break
                    else:
                        if token.lower() != value_token.lower():
                            match = False
                            break
                if match:
                    positions.append(i)
        return positions
    
    @staticmethod
    def calculate_net_worth(total_str: str, vat_str: str) -> str:
        try:
            total_val = float(total_str.replace(" ", "").replace(",","."))
            vat_val = float(vat_str.replace(" ","").replace(",","."))
            net_worth_val = total_val - vat_val
            net_worth_str = f"{net_worth_val:.2f}".replace(".",",")

            if " " in total_str and len(net_worth_str.split(",")[0]) > 3:
                integer_part, decimal_part = net_worth_str.split(",")
                spaced_integer = f"{integer_part[:-3]} {integer_part[-3:]}"
                net_worth_str = f"{spaced_integer},{decimal_part}"
            return net_worth_str
        except:
            return None
    
    @staticmethod
    def extract_summary_values_from_ocr(ocr_text:str) -> Dict[str,str]:
        summary_values = {}
        summary_start = ocr_text.upper().find("SUMMARY")
        if summary_start == -1:
            return summary_values
        summary_section = ocr_text[summary_start:]
        summary_pattern = r'10%\s+([\d\s,]+)\s+([\d\s,]+)\s+([\d\s,]+)'
        match = re.search(summary_pattern, summary_section)
        if match:
            summary_values['NET_WORTH'] = match.group(1).strip()
            summary_values['VAT'] = match.group(2).strip()
            summary_values['TOTAL'] = match.group(3).strip()
        
        return summary_values
    
    def annotate_invoice(self, ocr_text: str, json_data:Dict) -> Dict:
        tokens = ocr_text.split()
        labels = ["O"] * len(tokens)
        entities = {}

        invoice = json_data.get('invoice', {})

        entities["INVOICE_NUMBER"] = invoice.get("invoice_number")
        entities["INVOICE_DATE"] = invoice.get("invoice_date")
        entities["CLIENT_NAME"] = invoice.get("client_name")
        entities["SELLER_NAME"] = invoice.get("seller_name")
        entities["CLIENT_ADDRESS"] = invoice.get("client_address")
        entities["SELLER_ADDRESS"] = invoice.get("seller_address")
        entities["TAX_ID"] = invoice.get("tax_id") or invoice.get("seller_tax_id")

        subtotal = json_data.get("subtotal", {})
        entities["VAT"] = subtotal.get("tax") or subtotal.get("vat")
        entities["TOTAL"] = subtotal.get("total") or subtotal.get("gross_total")

        if entities["TOTAL"] and entities["VAT"]:
            calc = self.calculate_net_worth(entities["TOTAL"], entities["VAT"])
            if calc:
                entities["NET_WORTH"] = calc
        if "NET_WORTH" not in entities or not entities["NET_WORTH"]:
            entities.update(self.extract_summary_values_from_ocr(ocr_text))

        items = json_data.get("items", [])
        for idx, item in enumerate(items[:3]):
            entities[f"ITEM_DESC_{idx}"] = item.get("description")
            entities[f"QUANTITY_{idx}"] = item.get("quantity")
            entities[f"PRICE_{idx}"] = item.get("total_price")

        annotated_positions = set()
        for label, value in entities.items():
            if not value: continue
            clean_value = re.sub(r'\s+', ' ', str(value).strip())
            positions = self.find_entity_in_tokens(tokens, clean_value)
            for pos in positions:
                value_tokens = clean_value.split()
                if any(pos + j in annotated_positions for j in range(len(value_tokens))):
                    continue
                if pos + len(value_tokens) <= len(tokens):
                    base_label = label if not re.search(r'_\d+$', label) else '_'.join(label.split('_')[:-1])
                    labels[pos] = f"B-{base_label}"
                    for j in range(1, len(value_tokens)):
                        labels[pos + j] = f"I-{base_label}"
                    for j in range(len(value_tokens)):
                        annotated_positions.add(pos + j)
                    break
        return {"tokens": tokens, "ner_tags": labels}
    
    @staticmethod
    def validate_annotation(annotation: Dict) -> bool:
        if len(annotation["tokens"]) != len(annotation["ner_tags"]):
            return False
        for i, tag in enumerate(annotation["ner_tags"]):
            if tag.startswith("I-"):
                entity_type = tag[2:]
                if i == 0 or not (annotation["ner_tags"][i-1].endswith(entity_type)):
                    return False
        return True
    
    def process_csv_files(self, csv_files: List[str]) -> List[Dict]:
        for file in csv_files:
            print(f"Processing {file}...")
            try:
                df = pd.read_csv(file)
                for idx, row in df.iterrows():
                    try:
                        ocr_text = row["OCRed Text"]
                        json_data = json.loads(row["Json Data"]) if isinstance(row["Json Data"], str) else row["Json Data"]
                        annotation = self.annotate_invoice(ocr_text, json_data)
                        if self.validate_annotation(annotation):
                            annotation["id"] = self.id_counter
                            annotation["file_name"] = row.get("File Name", f"{file}_{idx}")
                            self.dataset.append(annotation)
                            self.id_counter += 1
                        else:
                            print(f"Invalid annotation in {file}, row {idx}")
                            self.error_count += 1
                    except Exception as e:
                        print(f"Error at file {file}, row {idx}: {e}")
                        self.error_count += 1
            except Exception as e:
                print(f"Error reading file {file}: {e}")
                self.error_count += 1
        print(f"Processing complete! Total samples: {len(self.dataset)}, Errors: {self.error_count}")
        return self.dataset

    def save_dataset(self, output_file: str = "invoice_ner_dataset_testing.jsonl"):
        with open(output_file, "w", encoding="utf-8") as f:
            for sample in self.dataset:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"Dataset saved to {output_file}")

    def analyze_dataset(self):
        total_tokens = sum(len(sample["tokens"]) for sample in self.dataset)
        all_tags = [tag for sample in self.dataset for tag in sample["ner_tags"]]
        tag_distribution = Counter(all_tags)
        entity_counts = Counter()
        for tag, count in tag_distribution.items():
            if tag.startswith("B-"):
                entity_counts[tag[2:]] = count
        print(f"DATASET ANALYSIS")
        print(f"Total samples: {len(self.dataset)}")
        print(f"Total tokens: {total_tokens}")
        print(f"Average tokens per sample: {total_tokens / len(self.dataset):.1f}")
        print(f"\nEntity distribution:")
        for entity, count in entity_counts.most_common():
            print(f"  {entity}: {count}")
        print(f"\nTag distribution:")
        for tag, count in tag_distribution.most_common():
            print(f"  {tag}: {count}")