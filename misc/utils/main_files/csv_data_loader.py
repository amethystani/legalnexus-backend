#!/usr/bin/env python3
"""
CSV Data Loader for Legal Case Classification Dataset
Converts CSV data (binary/ternary classification) to the format used by the knowledge graph pipeline
"""
import os
import csv
import json
import sys
import re
from typing import List, Dict
from langchain.schema import Document

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def clean_text(text: str) -> str:
    """Clean and normalize text data"""
    if not text:
        return ""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters that might cause issues
    text = text.strip()
    return text

def extract_metadata_with_llm(text: str, filename: str, use_llm: bool = True) -> Dict:
    """
    Extract metadata using intelligent LLM-based extraction
    Novel approach: Use Gemini to understand legal text structure and extract entities
    
    Args:
        text: Legal case text
        filename: Source filename for fallback extraction
        use_llm: Whether to use LLM (True) or fallback to pattern matching (False)
    """
    metadata = {
        'source': filename,
        'title': filename,
        'court': 'Unknown Court',
        'judgment_date': 'Unknown Date',
        'id': filename
    }
    
    # Quick filename-based extraction
    if '_HC_' in filename:
        parts = filename.split('_HC_')
        if parts:
            court_name = parts[0].replace('_', ' ')
            metadata['court'] = f"{court_name} High Court"
            if len(parts) > 1:
                year = parts[1].split('_')[0]
                if year.isdigit() and len(year) == 4:
                    metadata['judgment_date'] = year
    elif '_SC_' in filename or 'Supreme' in filename:
        metadata['court'] = 'Supreme Court of India'
    
    # Use first 2000 chars for analysis (sufficient for metadata)
    search_text = text[:2000]
    
    if use_llm:
        try:
            # Use LLM for intelligent extraction
            llm_metadata = _extract_with_gemini(search_text)
            
            # Merge LLM results with existing metadata
            if llm_metadata:
                for key, value in llm_metadata.items():
                    if value and value not in ['Unknown', 'Unknown Court', 'Unknown Date', 'None', None]:
                        metadata[key] = value
        except Exception as e:
            # Fallback to pattern matching if LLM fails
            print(f"LLM extraction failed, using pattern matching: {e}")
            use_llm = False
    
    if not use_llm:
        # Fallback: Pattern-based extraction
        metadata.update(_extract_with_patterns(search_text))
    
    # Extract title (always use text-based extraction)
    sentences = text.split('.')
    for sentence in sentences[:5]:
        clean = sentence.strip()
        if 20 < len(clean) < 200 and not clean.startswith('Ratio'):
            metadata['title'] = clean
            break
    
    return metadata


def _extract_with_gemini_batch(cases: List[tuple]) -> List[Dict]:
    """
    NOVEL: Batch extraction to avoid rate limits
    Process multiple cases in a single API call
    
    Args:
        cases: List of (text, filename) tuples
    
    Returns:
        List of metadata dicts for each case
    """
    try:
        import google.generativeai as genai
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        import json
        import time
        
        # Configure Gemini
        GOOGLE_API_KEY = "AIzaSyA0dLTfkzxcZYP6KidlFClAyMLl6mea1y8"
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # Try available models
        model_names = [
            'models/gemini-2.5-flash',
            'models/gemini-flash-lite-latest', 
            'models/gemini-2.5-flash-lite'
        ]
        
        model = None
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                break
            except:
                continue
        
        if not model:
            raise Exception("No compatible Gemini model available")
        
        # Build batch prompt with all cases
        batch_text = "You are a legal document parser. Extract metadata from these Indian court cases.\n\n"
        batch_text += "For EACH case, return a JSON object. Return a JSON array of objects.\n\n"
        
        for i, (text, filename) in enumerate(cases):
            # Use first 800 chars to avoid RECITATION (shorter = less likely to match training data)
            case_snippet = text[:800]
            batch_text += f"--- CASE {i} (ID: {filename}) ---\n{case_snippet}\n\n"
        
        batch_text += """
Extract for each case and return JSON array:
[
  {
    "case_id": "filename from CASE X header",
    "judges": ["judge names without titles like Hon'ble, Justice, Mr., Dr., J."],
    "date": "date if found",
    "court": "court name if mentioned",
    "statutes": ["Section X", "Article Y"],
    "acts": ["Act Name Year"]
  }
]

Rules:
- Return ONLY the JSON array, no other text
- Use [] for empty fields
- Extract what you clearly see, don't guess
- Keep judge names clean (remove titles)

JSON Array:"""
        
        # Generate with safety disabled
        response = model.generate_content(
            batch_text,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
            generation_config={
                'temperature': 0.1,
                'max_output_tokens': 2000,  # More tokens for batch
                'top_p': 0.95,
            }
        )
        
        # Get response text
        result_text = None
        try:
            result_text = response.text.strip()
        except:
            try:
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if candidate.content and candidate.content.parts:
                        result_text = candidate.content.parts[0].text.strip()
            except:
                pass
        
        if not result_text:
            if response.candidates and len(response.candidates) > 0:
                finish_reason = response.candidates[0].finish_reason
                if finish_reason == 2:
                    raise Exception("RECITATION filter - text too similar to training data")
            raise Exception("No valid response")
        
        # Clean JSON
        if '```' in result_text:
            parts = result_text.split('```')
            for part in parts:
                if '[' in part or 'json' in part.lower():
                    result_text = part.replace('json', '').strip()
                    break
        
        # Find JSON array
        start = result_text.find('[')
        end = result_text.rfind(']') + 1
        if start >= 0 and end > start:
            result_text = result_text[start:end]
        
        results = json.loads(result_text)
        
        # Clean and validate
        cleaned_results = []
        for result in results:
            cleaned = {}
            if result.get('judges') and isinstance(result['judges'], list):
                cleaned['judges'] = [j.strip() for j in result['judges'] if j and len(j) > 2][:5]
            if result.get('date') and result['date'] not in ['null', 'None', '']:
                cleaned['judgment_date'] = result['date']
            if result.get('court') and result['court'] not in ['null', 'None', '']:
                cleaned['court'] = result['court']
            if result.get('statutes') and isinstance(result['statutes'], list):
                cleaned['statutes'] = [s.strip() for s in result['statutes'] if s and len(s) > 3][:15]
            if result.get('acts') and isinstance(result['acts'], list):
                cleaned['acts'] = [a.strip() for a in result['acts'] if a and len(a) > 3][:10]
            cleaned_results.append(cleaned)
        
        return cleaned_results
        
    except Exception as e:
        print(f"Batch extraction error: {e}")
        # Return empty dicts for fallback to pattern matching
        return [{} for _ in cases]


def _extract_with_gemini(text: str) -> Dict:
    """
    Single case extraction (kept for backwards compatibility)
    Now uses batch with size 1
    """
    results = _extract_with_gemini_batch([(text, "single")])
    return results[0] if results else {}


def _extract_with_patterns(text: str) -> Dict:
    """Fallback pattern-based extraction (fast but less accurate)"""
    metadata = {}
    
    # Judge patterns
    judge_patterns = [
        r"Hon(?:'|')ble\s+(?:Mr\.|Dr\.|Justice|Judge)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})",
        r'(?:Justice|Judge)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
        r'Coram\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})',
    ]
    judges = []
    for pattern in judge_patterns:
        judges.extend(re.findall(pattern, text))
    if judges:
        exclude = {'This', 'The', 'Court', 'Honble', 'Justice', 'Judge'}
        cleaned = [j.strip() for j in judges if len(j) > 3 and j not in exclude]
        if cleaned:
            metadata['judges'] = list(set(cleaned))[:5]
    
    # Date patterns
    date_patterns = [
        r'\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b',
        r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b',
    ]
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            metadata['judgment_date'] = matches[0]
            break
    
    # Statutes
    statute_patterns = [
        r'Section\s+\d+[A-Z]*(?:\s*\([A-Za-z0-9]+\))?',
        r'Article\s+\d+[A-Z]*',
    ]
    statutes = []
    for pattern in statute_patterns:
        statutes.extend(re.findall(pattern, text))
    if statutes:
        metadata['statutes'] = list(set(statutes))[:15]
    
    # Acts
    act_patterns = [
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4}\s+Act\s*(?:\d{4})?)',
        r'(I\.?P\.?C\.?)',
        r'(Cr\.?P\.?C\.?)',
    ]
    acts = []
    for pattern in act_patterns:
        acts.extend(re.findall(pattern, text))
    if acts:
        metadata['acts'] = list(set([a.strip() for a in acts if len(a) > 3]))[:10]
    
    return metadata


def extract_metadata_from_text(text: str, filename: str) -> Dict:
    """
    Main extraction function - uses pattern-based extraction (fast, no API limits)
    LLM extraction disabled to avoid daily quota (250 req/day)
    """
    return extract_metadata_with_llm(text, filename, use_llm=False)

def load_csv_data(csv_file_path: str, max_rows: int = None, batch_size: int = 5) -> List[Document]:
    """
    Load legal case data from CSV file with BATCH LLM extraction
    
    Args:
        csv_file_path: Path to CSV file
        max_rows: Maximum rows to process
        batch_size: Number of cases to process in one API call (default 5)
    
    Returns:
        List of Document objects
    """
    docs = []
    
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found: {csv_file_path}")
        return docs
    
    print(f"Loading data from: {csv_file_path}")
    
    # Increase CSV field size limit to handle large legal case texts
    csv.field_size_limit(10 * 1024 * 1024)  # 10MB limit
    
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Collect cases for batch processing
            case_batch = []
            row_data = []
            
            for i, row in enumerate(reader):
                if max_rows and i >= max_rows:
                    break
                
                # Extract fields
                filename = row.get('filename', f'case_{i}')
                text = row.get('text', '')
                label = row.get('label', '0')
                
                # Clean text
                text = clean_text(text)
                
                if not text or len(text) < 50:
                    continue
                
                # Add to batch
                case_batch.append((text, filename))
                row_data.append({'text': text, 'filename': filename, 'label': label})
                
                # Process batch when full
                if len(case_batch) >= batch_size:
                    # PATTERN-ONLY EXTRACTION (LLM disabled due to quota)
                    # llm_results = _extract_with_gemini_batch(case_batch)
                    
                    # Create documents with extracted metadata
                    for j, data in enumerate(row_data):
                        metadata = extract_metadata_from_text(data['text'], data['filename'])
                        
                        # LLM extraction disabled
                        # if j < len(llm_results) and llm_results[j]:
                        #     for key, value in llm_results[j].items():
                        #         if value and value not in ['Unknown', 'Unknown Court', 'Unknown Date']:
                        #             metadata[key] = value
                        
                        metadata['classification_label'] = data['label']
                        metadata['id'] = f"{data['filename']}_{i}"
                        
                        doc = Document(
                            page_content=data['text'],
                            metadata=metadata
                        )
                        docs.append(doc)
                    
                    # Clear batch
                    case_batch = []
                    row_data = []
                    
                    # Progress
                    if (i + 1) % 100 == 0:
                        print(f"Processed {i + 1} cases...")
            
            # Process remaining cases in final batch
            if case_batch:
                # PATTERN-ONLY EXTRACTION (LLM disabled due to quota)
                # llm_results = _extract_with_gemini_batch(case_batch)
                
                for j, data in enumerate(row_data):
                    metadata = extract_metadata_from_text(data['text'], data['filename'])
                    
                    # LLM extraction disabled
                    # if j < len(llm_results) and llm_results[j]:
                    #     for key, value in llm_results[j].items():
                    #         if value and value not in ['Unknown', 'Unknown Court', 'Unknown Date']:
                    #             metadata[key] = value
                    
                    metadata['classification_label'] = data['label']
                    metadata['id'] = f"{data['filename']}_{len(docs)}"
                    
                    doc = Document(
                        page_content=data['text'],
                        metadata=metadata
                    )
                    docs.append(doc)
        
        print(f"Successfully loaded {len(docs)} legal cases from CSV")
        return docs
    
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return docs

def convert_csv_to_json(csv_file_path: str, output_dir: str, max_rows: int = None):
    """
    Convert CSV data to JSON format used by the knowledge graph
    
    Args:
        csv_file_path: Path to input CSV file
        output_dir: Directory to save JSON files
        max_rows: Maximum number of rows to convert (None for all)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    docs = load_csv_data(csv_file_path, max_rows)
    
    print(f"\nConverting {len(docs)} cases to JSON format...")
    
    for i, doc in enumerate(docs):
        # Create JSON structure
        case_data = {
            'id': doc.metadata.get('id', f'case_{i}'),
            'title': doc.metadata.get('title', 'Unknown Case'),
            'court': doc.metadata.get('court', 'Unknown Court'),
            'judgment_date': doc.metadata.get('judgment_date', 'Unknown Date'),
            'source': doc.metadata.get('source', ''),
            'content': doc.page_content,
            'metadata': {
                'classification_label': doc.metadata.get('classification_label', '0')
            }
        }
        
        # Add judges if available
        if 'judges' in doc.metadata:
            case_data['entities'] = {
                'judges': doc.metadata['judges']
            }
        
        # Add statutes if available
        if 'statutes' in doc.metadata:
            if 'entities' not in case_data:
                case_data['entities'] = {}
            case_data['entities']['statutes'] = doc.metadata['statutes']
        
        # Save to JSON file
        json_filename = f"{doc.metadata.get('source', f'case_{i}').replace('.csv', '')}.json"
        json_path = os.path.join(output_dir, json_filename)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(case_data, f, ensure_ascii=False, indent=2)
        
        if (i + 1) % 100 == 0:
            print(f"Converted {i + 1} cases to JSON...")
    
    print(f"\nSuccessfully converted {len(docs)} cases to JSON format")
    print(f"Output directory: {output_dir}")

def load_all_csv_data(data_dir: str = "data", max_cases_per_file: int = 100) -> List[Document]:
    """
    Load all CSV data from binary and ternary classification datasets
    
    Args:
        data_dir: Base data directory
        max_cases_per_file: Maximum cases to load from each CSV file
    
    Returns:
        Combined list of Document objects
    """
    all_docs = []
    
    # Binary classification data
    binary_csv = os.path.join(data_dir, "binary_dev", "CJPE_ext_SCI_HCs_Tribunals_daily_orders_dev.csv")
    if os.path.exists(binary_csv):
        print("\n=== Loading Binary Classification Data ===")
        binary_docs = load_csv_data(binary_csv, max_cases_per_file)
        all_docs.extend(binary_docs)
        print(f"Loaded {len(binary_docs)} cases from binary dataset")
    
    # Ternary classification data
    ternary_csv = os.path.join(data_dir, "ternary_dev", "CJPE_ext_SCI_HCs_tribunals_dailyorder_dev_wo_RoD_ternary.csv")
    if os.path.exists(ternary_csv):
        print("\n=== Loading Ternary Classification Data ===")
        ternary_docs = load_csv_data(ternary_csv, max_cases_per_file)
        all_docs.extend(ternary_docs)
        print(f"Loaded {len(ternary_docs)} cases from ternary dataset")
    
    print(f"\n=== Total: {len(all_docs)} cases loaded ===")
    return all_docs

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CSV Data Loader for Legal Cases')
    parser.add_argument('--convert', action='store_true', help='Convert CSV to JSON format')
    parser.add_argument('--input', type=str, help='Input CSV file path')
    parser.add_argument('--output', type=str, default='data/converted_cases', help='Output directory for JSON files')
    parser.add_argument('--max-rows', type=int, default=100, help='Maximum rows to process')
    parser.add_argument('--test', action='store_true', help='Test loading data')
    
    args = parser.parse_args()
    
    if args.convert:
        if not args.input:
            print("Error: --input required when using --convert")
            sys.exit(1)
        convert_csv_to_json(args.input, args.output, args.max_rows)
    elif args.test:
        # Test loading
        print("Testing data loading...")
        docs = load_all_csv_data(max_cases_per_file=5)
        print(f"\nSample case:")
        if docs:
            print(f"Title: {docs[0].metadata.get('title')}")
            print(f"Court: {docs[0].metadata.get('court')}")
            print(f"Text length: {len(docs[0].page_content)} characters")
            print(f"First 200 chars: {docs[0].page_content[:200]}...")
    else:
        parser.print_help()

