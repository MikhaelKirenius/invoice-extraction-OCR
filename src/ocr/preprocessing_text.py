import re 
from transformers import pipeline
import numpy as np 
import os

class TextProcessingNER:
    def __init__(self, model, tokenizer):
        self.ner_pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="max"
        )
        
    def merge_subword_tokens(self, entities):
        """Merge subword tokens (##) back together"""
        merged_entities = []
        
        i = 0
        while i < len(entities):
            entity = entities[i]
            
            if entity['entity_group'] in ['INVOICE_NUMBER', 'INVOICE_DATE', 'PRICE', 'TOTAL', 'VAT', 'NET_WORTH']:
                complete_word = entity['word'].replace('##', '')
                complete_start = entity['start']
                complete_end = entity['end']
                total_score = entity['score']
                count = 1
                
                j = i + 1
                while j < len(entities):
                    next_entity = entities[j]
                    
                    if (next_entity['entity_group'] == entity['entity_group'] and
                        next_entity['start'] <= complete_end + 2):  
                        
                        next_word = next_entity['word'].replace('##', '')
                        complete_word += next_word
                        complete_end = next_entity['end']
                        total_score += next_entity['score']
                        count += 1
                        j += 1
                    else:
                        break
                
                merged_entities.append({
                    'entity_group': entity['entity_group'],
                    'score': total_score / count,  
                    'word': complete_word,
                    'start': complete_start,
                    'end': complete_end
                })
                
                i = j
            else:
                merged_entities.append(entity)
                i += 1
        
        return merged_entities

    def _pick_rightmost_amount(self, line: str):
        nums = re.findall(r'(\d[\d\s.,]*\d)', line)
        return nums[-1].strip() if nums else None

    def _extract_total_from_summary(self, text: str):
        total_lines = re.findall(r'(?im)^.*total.*$', text)
        for line in reversed(total_lines):
            val = self._pick_rightmost_amount(line)
            if val:
                return val
        gross_lines = re.findall(r'(?im)^.*gross\s*worth.*$', text)
        for line in reversed(gross_lines):
            val = self._pick_rightmost_amount(line)
            if val:
                return val
        return self._pick_rightmost_amount(text)
    
    def regex_extraction(self, text):
        """Fallback regex extraction for critical entities"""
        regex_entities = {}
        
        invoice_patterns = [
            r'Invoice\s+no:?\s*(\d+)',
            r'Invoice\s+number:?\s*(\d+)',
            r'#\s*(\d+)'
        ]
        
        for pattern in invoice_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                regex_entities['INVOICE_NUMBER'] = match.group(1)
                break
        
        date_patterns = [
            r'Date\s+of\s+issue:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4})',
            r'Date:?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4})',
            r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                regex_entities['INVOICE_DATE'] = match.group(1)
                break
        
        ## Error Disabled for TOTAL extraction
        # total_patterns = [
        #     r'Total\s+\$?\s*(\d+[\s,]*\d*[.,]?\d*)',
        #     r'\$\s*(\d+[\s,]+\d+[.,]\d+)(?=\s*$|\s*\n|\s*[A-Z])',  
        #     r'(\d+[\s,]+\d+[.,]\d+)(?=\s*$)',  
        # ]
        
        # total_matches = []
        # for pattern in total_patterns:
        #     matches = re.findall(pattern, text, re.IGNORECASE)
        #     total_matches.extend(matches)
        
        # if total_matches:
        #     final_total = total_matches[-1]
        #     final_total = re.sub(r'\s+', ' ', final_total)  
        #     regex_entities['TOTAL'] = final_total
        
        ## New Methods
        total_value = self._extract_total_from_summary(text)
        if total_value:
            cleaned = total_value.replace('\u00A0', ' ').strip()
            cleaned = re.sub(r'[^\d.,\s]', '', cleaned)
            cleaned = cleaned.replace(' ', '')
            if cleaned.count(',') == 1 and cleaned.count('.') == 0:
                cleaned = cleaned.replace(',', '.')
            regex_entities['TOTAL'] = total_value

        
        seller_match = re.search(r'Seller:?\s*([A-Za-z\s\-&,]+?)(?=\s+Client|\s+\d|\s*$)', text, re.IGNORECASE)
        if seller_match:
            regex_entities['SELLER_NAME'] = seller_match.group(1).strip()
        
        client_match = re.search(r'Client:?\s*([A-Za-z\s\-&]+?)(?=\s+\d|\s+Tax|\s*$)', text, re.IGNORECASE)
        if client_match:
            regex_entities['CLIENT_NAME'] = client_match.group(1).strip()
        
        return regex_entities
    
    def restore_punctuation(self, text, entity_text, start_pos, end_pos):
        """Restore punctuation between words by checking the original text"""
        if not entity_text or ' ' not in entity_text:
            return entity_text
        
        window_start = max(0, start_pos - 5)
        window_end = min(len(text), end_pos + 5)
        window_text = text[window_start:window_end]
        
        words = entity_text.split()
        if len(words) < 2:
            return entity_text
        
        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i + 1]
            
            pattern = re.escape(current_word) + r'([,\-\s]+)' + re.escape(next_word)
            match = re.search(pattern, window_text, re.IGNORECASE)
            
            if match:
                separator = match.group(1)
                if ',' in separator:
                    words[i] = current_word + ','
        
        return ' '.join(words)
    
    def extract_entities(self, text):
        """Main extraction method combining NER and regex"""
        try:
            ner_results = self.ner_pipeline(text)
            merged_results = self.merge_subword_tokens(ner_results)
            
            ner_entities = {}
            for entity in merged_results:
                entity_type = entity['entity_group']
                if entity_type not in ner_entities:
                    ner_entities[entity_type] = []
                ner_entities[entity_type].append(entity)
            
        except Exception as e:
            print(f"NER model failed: {e}")
            ner_entities = {}

        regex_entities = self.regex_extraction(text)
        
        final_entities = {}
        
        for entity_type in ['INVOICE_NUMBER', 'INVOICE_DATE', 'SELLER_NAME', 'CLIENT_NAME', 'TOTAL']:
            
            ner_candidates = ner_entities.get(entity_type, [])
            regex_candidate = regex_entities.get(entity_type)
            
            if entity_type == 'INVOICE_NUMBER':
                best_ner = None
                for candidate in ner_candidates:
                    if candidate['word'].replace('##', '').replace(' ', '').isdigit():
                        if not best_ner or candidate['score'] > best_ner['score']:
                            best_ner = candidate
                
                if best_ner and best_ner['score'] > 0.8:
                    final_entities[entity_type] = best_ner['word'].replace('##', '').replace(' ', '')
                elif regex_candidate:
                    final_entities[entity_type] = regex_candidate
                    
            elif entity_type == 'INVOICE_DATE':
                if regex_candidate and len(regex_candidate) >= 8:  
                    final_entities[entity_type] = regex_candidate
                elif ner_candidates:
                    date_parts = [c['word'] for c in ner_candidates]
                    reconstructed = ''.join(date_parts)
                    if len(reconstructed) >= 6:
                        final_entities[entity_type] = reconstructed
                        
            elif entity_type == 'TOTAL':
                if regex_candidate:
                    final_entities[entity_type] = regex_candidate
                elif ner_candidates:
                    best_total = max(ner_candidates, key=lambda x: x['score'])
                    final_entities[entity_type] = best_total['word']
                    
            else:  
                if ner_candidates:
                    good_candidates = [c for c in ner_candidates if c['score'] > 0.8]
                    
                    if good_candidates:
                        if len(good_candidates) == 1:
                            final_entities[entity_type] = good_candidates[0]['word']
                        else:
                            good_candidates.sort(key=lambda x: x['start'])
                            
                            if entity_type == 'CLIENT_NAME' and len(good_candidates) > 1:
                                first_candidate = good_candidates[0]
                                last_candidate = good_candidates[-1]
                                gap = last_candidate['start'] - first_candidate['end']
                                
                                if gap > 50:  
                                    final_entities[entity_type] = last_candidate['word']
                                else:
                                    combined_name = first_candidate['word']
                                    last_end = first_candidate['end']
                                    
                                    for candidate in good_candidates[1:]:
                                        gap = candidate['start'] - last_end
                                        if gap <= 20: 
                                            if gap > 1:
                                                combined_name += " " + candidate['word']
                                            else:
                                                combined_name += candidate['word']
                                            last_end = candidate['end']
                                        else:
                                            break
                                    
                                    final_entities[entity_type] = combined_name.strip()
                            else:
                                combined_name = good_candidates[0]['word']
                                last_end = good_candidates[0]['end']
                                
                                for candidate in good_candidates[1:]:
                                    gap = candidate['start'] - last_end
                                    if gap <= 20:  
                                        if gap > 1:
                                            combined_name += " " + candidate['word']
                                        else:
                                            combined_name += candidate['word']
                                        last_end = candidate['end']
                                    else:
                                        break
                                
                                final_entities[entity_type] = combined_name.strip()
                                
                    elif regex_candidate:
                        final_entities[entity_type] = regex_candidate
                elif regex_candidate:
                    final_entities[entity_type] = regex_candidate
        
        for entity_type in ['SELLER_NAME', 'CLIENT_NAME']:
            if entity_type in final_entities:
                original_candidates = ner_entities.get(entity_type, [])
                if original_candidates:
                    start_pos = min(c['start'] for c in original_candidates)
                    end_pos = max(c['end'] for c in original_candidates)
                    
                    final_entities[entity_type] = self.restore_punctuation(
                        text, 
                        final_entities[entity_type], 
                        start_pos, 
                        end_pos
                    )
        
        return final_entities
    


    

