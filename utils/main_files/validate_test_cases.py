import os
import json
import glob
import sys

def validate_test_cases(test_cases_dir="data/test_cases"):
    """
    Validate that test cases have the correct format
    """
    # Adjust path if running from Backend directory
    if os.path.basename(os.getcwd()) == "Backend":
        test_cases_dir = os.path.join("..", test_cases_dir)
    
    print(f"Validating test cases in {os.path.abspath(test_cases_dir)}")
    
    if not os.path.exists(test_cases_dir):
        print(f"ERROR: Test cases directory {test_cases_dir} not found!")
        return False
    
    test_case_files = glob.glob(os.path.join(test_cases_dir, "*.json"))
    
    if not test_case_files:
        print(f"ERROR: No test case files found in {test_cases_dir}")
        return False
    
    print(f"Found {len(test_case_files)} test case files:")
    
    all_valid = True
    required_fields = ['content', 'title', 'court', 'judgment_date', 'id']
    
    for file_path in test_case_files:
        filename = os.path.basename(file_path)
        print(f"\nValidating {filename}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Check required fields
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    print(f"  ERROR: Missing required fields: {', '.join(missing_fields)}")
                    all_valid = False
                else:
                    print("  ✓ All required fields present")
                
                # Check content length
                if 'content' in data:
                    content_length = len(data['content'])
                    if content_length < 100:
                        print(f"  WARNING: Content is very short ({content_length} chars)")
                    else:
                        print(f"  ✓ Content length: {content_length} chars")
                
                # Validate entities structure if present
                if 'entities' in data:
                    if not isinstance(data['entities'], dict):
                        print("  ERROR: 'entities' should be a dictionary")
                        all_valid = False
                    else:
                        print(f"  ✓ Entities structure valid with {len(data['entities'])} entity types")
                        
                        # Check specific entities
                        if 'judges' in data['entities']:
                            if isinstance(data['entities']['judges'], list):
                                print(f"  ✓ Found {len(data['entities']['judges'])} judges")
                            else:
                                print("  ERROR: 'judges' should be a list")
                                all_valid = False
                        
                        if 'statutes' in data['entities']:
                            if isinstance(data['entities']['statutes'], list):
                                print(f"  ✓ Found {len(data['entities']['statutes'])} statutes")
                            else:
                                print("  ERROR: 'statutes' should be a list")
                                all_valid = False
                else:
                    print("  WARNING: No 'entities' field found")
                
                # Validate metadata structure if present
                if 'metadata' in data:
                    if not isinstance(data['metadata'], dict):
                        print("  ERROR: 'metadata' should be a dictionary")
                        all_valid = False
                    else:
                        print(f"  ✓ Metadata structure valid with {len(data['metadata'])} fields")
                
        except json.JSONDecodeError:
            print(f"  ERROR: Invalid JSON format in {filename}")
            all_valid = False
        except Exception as e:
            print(f"  ERROR: Failed to validate {filename}: {str(e)}")
            all_valid = False
    
    if all_valid:
        print("\nAll test cases are valid!")
    else:
        print("\nSome test cases have validation errors. Please fix them.")
    
    return all_valid

if __name__ == "__main__":
    # Get directory from command line if provided
    test_cases_dir = sys.argv[1] if len(sys.argv) > 1 else "data/test_cases"
    validate_test_cases(test_cases_dir) 