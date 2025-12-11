import json
import os
import argparse
from datetime import datetime

def label_studio_to_json(label_studio_file, output_file=None):
    """
    Convert Label Studio annotation format to our JSON format
    
    Args:
        label_studio_file (str): Path to Label Studio JSON export file
        output_file (str): Path to output JSON file (optional)
    
    Returns:
        dict: Converted data in our JSON format
    """
    with open(label_studio_file, 'r') as f:
        label_studio_data = json.load(f)
    
    # Initialize our JSON structure
    case_data = {
        "id": "",
        "source": "",
        "title": "",
        "court": "",
        "judgment_date": "",
        "content": "",
        "entities": {
            "cases": [],
            "statutes": [],
            "judges": [],
            "jurisdictions": []
        },
        "metadata": {
            "judges": "",
            "primary_case_title": "",
            "primary_case_number": "",
            "primary_citation": ""
        },
        "cited_cases": [],
        "final_decision": "",
        "case_type": "",
        "created_at": datetime.now().isoformat()
    }
    
    # Process each annotated item
    for item in label_studio_data:
        # Extract task data (original content)
        task_data = item.get('data', {})
        case_data['content'] = task_data.get('content', '')
        
        # Extract annotations
        annotations = item.get('annotations', [])
        if not annotations:
            continue
            
        result = annotations[0].get('result', [])
        
        # Process each annotation result
        for annotation in result:
            value = annotation.get('value', {})
            
            # Process basic text fields
            if annotation.get('type') == 'textarea':
                field_name = annotation.get('from_name')
                
                if field_name == 'title':
                    case_data['title'] = value.get('text', [''])[0]
                elif field_name == 'court':
                    case_data['court'] = value.get('text', [''])[0]
                elif field_name == 'case_type':
                    case_data['case_type'] = value.get('text', [''])[0]
                elif field_name == 'final_decision':
                    case_data['final_decision'] = value.get('text', [''])[0]
                elif field_name == 'source':
                    case_data['source'] = value.get('text', [''])[0]
                elif field_name == 'statutes':
                    statutes_text = value.get('text', [''])[0]
                    case_data['entities']['statutes'] = [s.strip() for s in statutes_text.split('\n') if s.strip()]
                elif field_name == 'judges':
                    judges_text = value.get('text', [''])[0]
                    case_data['entities']['judges'] = [j.strip() for j in judges_text.split('\n') if j.strip()]
                    case_data['metadata']['judges'] = ', '.join(case_data['entities']['judges'])
                elif field_name == 'jurisdictions':
                    jurisdictions_text = value.get('text', [''])[0]
                    case_data['entities']['jurisdictions'] = [j.strip() for j in jurisdictions_text.split('\n') if j.strip()]
                elif field_name == 'primary_case_title':
                    case_data['metadata']['primary_case_title'] = value.get('text', [''])[0]
                elif field_name == 'primary_case_number':
                    case_data['metadata']['primary_case_number'] = value.get('text', [''])[0]
                elif field_name == 'primary_citation':
                    case_data['metadata']['primary_citation'] = value.get('text', [''])[0]
                # Handle cited case fields
                elif field_name.startswith('cited_case_title_'):
                    idx = field_name.split('_')[-1]
                    # Make sure we have the corresponding cited case in the list
                    while len(case_data['cited_cases']) <= int(idx):
                        case_data['cited_cases'].append({
                            "title": "",
                            "citation": "",
                            "relevance": "Medium"
                        })
                    case_data['cited_cases'][int(idx)]['title'] = value.get('text', [''])[0]
                    
                    # Also add to entities.cases if not already there
                    title = value.get('text', [''])[0]
                    if title and title not in case_data['entities']['cases']:
                        case_data['entities']['cases'].append(title)
                        
                elif field_name.startswith('cited_case_citation_'):
                    idx = field_name.split('_')[-1]
                    # Make sure we have the corresponding cited case in the list
                    while len(case_data['cited_cases']) <= int(idx):
                        case_data['cited_cases'].append({
                            "title": "",
                            "citation": "",
                            "relevance": "Medium"
                        })
                    case_data['cited_cases'][int(idx)]['citation'] = value.get('text', [''])[0]
            
            # Process date fields
            elif annotation.get('type') == 'datetime':
                field_name = annotation.get('from_name')
                if field_name == 'judgment_date':
                    case_data['judgment_date'] = value.get('datetime', '')
            
            # Process choice fields
            elif annotation.get('type') == 'choices':
                field_name = annotation.get('from_name')
                if field_name.startswith('cited_case_relevance_'):
                    idx = field_name.split('_')[-1]
                    # Make sure we have the corresponding cited case in the list
                    while len(case_data['cited_cases']) <= int(idx):
                        case_data['cited_cases'].append({
                            "title": "",
                            "citation": "",
                            "relevance": "Medium"
                        })
                    choices = value.get('choices', [])
                    if choices:
                        case_data['cited_cases'][int(idx)]['relevance'] = choices[0]
            
            # Process labeled entities
            elif annotation.get('type') == 'labels':
                field_name = annotation.get('from_name')
                if field_name == 'entities':
                    labels = value.get('labels', [])
                    text = value.get('text', '')
                    
                    if 'Case' in labels and text not in case_data['entities']['cases']:
                        case_data['entities']['cases'].append(text)
                    elif 'Statute' in labels and text not in case_data['entities']['statutes']:
                        case_data['entities']['statutes'].append(text)
                    elif 'Judge' in labels and text not in case_data['entities']['judges']:
                        case_data['entities']['judges'].append(text)
                        case_data['metadata']['judges'] = ', '.join(case_data['entities']['judges'])
                    elif 'Jurisdiction' in labels and text not in case_data['entities']['jurisdictions']:
                        case_data['entities']['jurisdictions'].append(text)
    
    # Generate ID from title if not present
    if not case_data['id'] and case_data['title']:
        case_data['id'] = case_data['title'].lower().replace(' ', '_')
    
    # Clean up the cited_cases list by removing empty entries
    case_data['cited_cases'] = [case for case in case_data['cited_cases'] if case['title']]
    
    # Write to file if output_file is provided
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(case_data, f, indent=2)
            
    return case_data

def json_to_label_studio(json_file, output_file=None):
    """
    Convert our JSON format to Label Studio import format
    
    Args:
        json_file (str): Path to our JSON file
        output_file (str): Path to output Label Studio JSON file (optional)
    
    Returns:
        list: Converted data in Label Studio format
    """
    with open(json_file, 'r') as f:
        case_data = json.load(f)
    
    # Create Label Studio task format
    label_studio_data = [{
        "data": {
            "content": case_data.get('content', '')
        },
        "annotations": [
            {
                "id": 1,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "lead_time": 0,
                "result": []
            }
        ]
    }]
    
    result = label_studio_data[0]['annotations'][0]['result']
    
    # Add basic fields
    result.append({
        "id": "title",
        "type": "textarea",
        "value": {
            "text": [case_data.get('title', '')]
        },
        "to_name": "text",
        "from_name": "title"
    })
    
    result.append({
        "id": "court",
        "type": "textarea",
        "value": {
            "text": [case_data.get('court', '')]
        },
        "to_name": "text",
        "from_name": "court"
    })
    
    result.append({
        "id": "case_type",
        "type": "textarea",
        "value": {
            "text": [case_data.get('case_type', '')]
        },
        "to_name": "text",
        "from_name": "case_type"
    })
    
    result.append({
        "id": "judgment_date",
        "type": "datetime",
        "value": {
            "datetime": case_data.get('judgment_date', '')
        },
        "to_name": "text",
        "from_name": "judgment_date"
    })
    
    result.append({
        "id": "final_decision",
        "type": "textarea",
        "value": {
            "text": [case_data.get('final_decision', '')]
        },
        "to_name": "text",
        "from_name": "final_decision"
    })
    
    result.append({
        "id": "source",
        "type": "textarea",
        "value": {
            "text": [case_data.get('source', '')]
        },
        "to_name": "text",
        "from_name": "source"
    })
    
    # Add entities
    entities = case_data.get('entities', {})
    
    # Add statutes as text area
    statutes_text = '\n'.join(entities.get('statutes', []))
    result.append({
        "id": "statutes",
        "type": "textarea",
        "value": {
            "text": [statutes_text]
        },
        "to_name": "text",
        "from_name": "statutes"
    })
    
    # Add judges as text area
    judges_text = '\n'.join(entities.get('judges', []))
    result.append({
        "id": "judges",
        "type": "textarea",
        "value": {
            "text": [judges_text]
        },
        "to_name": "text",
        "from_name": "judges"
    })
    
    # Add jurisdictions as text area
    jurisdictions_text = '\n'.join(entities.get('jurisdictions', []))
    result.append({
        "id": "jurisdictions",
        "type": "textarea",
        "value": {
            "text": [jurisdictions_text]
        },
        "to_name": "text",
        "from_name": "jurisdictions"
    })
    
    # Add metadata
    metadata = case_data.get('metadata', {})
    result.append({
        "id": "primary_case_title",
        "type": "textarea",
        "value": {
            "text": [metadata.get('primary_case_title', '')]
        },
        "to_name": "text",
        "from_name": "primary_case_title"
    })
    
    result.append({
        "id": "primary_case_number",
        "type": "textarea",
        "value": {
            "text": [metadata.get('primary_case_number', '')]
        },
        "to_name": "text",
        "from_name": "primary_case_number"
    })
    
    result.append({
        "id": "primary_citation",
        "type": "textarea",
        "value": {
            "text": [metadata.get('primary_citation', '')]
        },
        "to_name": "text",
        "from_name": "primary_citation"
    })
    
    # Add cited cases
    cited_cases = case_data.get('cited_cases', [])
    for idx, cited_case in enumerate(cited_cases):
        # Add title
        result.append({
            "id": f"cited_case_title_{idx}",
            "type": "textarea",
            "value": {
                "text": [cited_case.get('title', '')]
            },
            "to_name": "text",
            "from_name": f"cited_case_title_{idx}"
        })
        
        # Add citation
        result.append({
            "id": f"cited_case_citation_{idx}",
            "type": "textarea",
            "value": {
                "text": [cited_case.get('citation', '')]
            },
            "to_name": "text",
            "from_name": f"cited_case_citation_{idx}"
        })
        
        # Add relevance
        result.append({
            "id": f"cited_case_relevance_{idx}",
            "type": "choices",
            "value": {
                "choices": [cited_case.get('relevance', 'Medium')]
            },
            "to_name": "text",
            "from_name": f"cited_case_relevance_{idx}"
        })
    
    # Add labeled entities
    for case_name in entities.get('cases', []):
        # Find the case name in content
        if case_name in case_data.get('content', ''):
            start = case_data['content'].find(case_name)
            if start >= 0:
                end = start + len(case_name)
                result.append({
                    "id": f"entity_case_{len(result)}",
                    "type": "labels",
                    "value": {
                        "start": start,
                        "end": end,
                        "text": case_name,
                        "labels": ["Case"]
                    },
                    "to_name": "text",
                    "from_name": "entities"
                })
    
    for statute in entities.get('statutes', []):
        # Find the statute in content
        if statute in case_data.get('content', ''):
            start = case_data['content'].find(statute)
            if start >= 0:
                end = start + len(statute)
                result.append({
                    "id": f"entity_statute_{len(result)}",
                    "type": "labels",
                    "value": {
                        "start": start,
                        "end": end,
                        "text": statute,
                        "labels": ["Statute"]
                    },
                    "to_name": "text",
                    "from_name": "entities"
                })
    
    for judge in entities.get('judges', []):
        # Find the judge in content
        if judge in case_data.get('content', ''):
            start = case_data['content'].find(judge)
            if start >= 0:
                end = start + len(judge)
                result.append({
                    "id": f"entity_judge_{len(result)}",
                    "type": "labels",
                    "value": {
                        "start": start,
                        "end": end,
                        "text": judge,
                        "labels": ["Judge"]
                    },
                    "to_name": "text",
                    "from_name": "entities"
                })
    
    for jurisdiction in entities.get('jurisdictions', []):
        # Find the jurisdiction in content
        if jurisdiction in case_data.get('content', ''):
            start = case_data['content'].find(jurisdiction)
            if start >= 0:
                end = start + len(jurisdiction)
                result.append({
                    "id": f"entity_jurisdiction_{len(result)}",
                    "type": "labels",
                    "value": {
                        "start": start,
                        "end": end,
                        "text": jurisdiction,
                        "labels": ["Jurisdiction"]
                    },
                    "to_name": "text",
                    "from_name": "entities"
                })
    
    # Write to file if output_file is provided
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(label_studio_data, f, indent=2)
            
    return label_studio_data

def main():
    parser = argparse.ArgumentParser(description='Convert between Label Studio and our JSON format')
    parser.add_argument('--input', '-i', required=True, help='Input file path')
    parser.add_argument('--output', '-o', required=True, help='Output file path')
    parser.add_argument('--direction', '-d', required=True, choices=['to_json', 'to_label_studio'], 
                        help='Conversion direction')
    
    args = parser.parse_args()
    
    if args.direction == 'to_json':
        label_studio_to_json(args.input, args.output)
        print(f"Converted {args.input} to {args.output}")
    else:
        json_to_label_studio(args.input, args.output)
        print(f"Converted {args.input} to {args.output}")

if __name__ == "__main__":
    main() 