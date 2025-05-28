#!/usr/bin/env python3
import os
import sys
import subprocess
import importlib.util

def run_command(cmd, cwd=None):
    """Run a shell command and return exit code"""
    print(f"\n{'='*80}")
    print(f"Running: {cmd}")
    print(f"{'='*80}")
    
    process = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        text=True
    )
    return process.returncode

def import_and_run(module_path, function_name='main'):
    """Import a Python module and run a specific function from it"""
    print(f"\n{'='*80}")
    print(f"Importing and running: {module_path} -> {function_name}()")
    print(f"{'='*80}")
    
    try:
        # Get the absolute path
        abs_path = os.path.abspath(module_path)
        
        # Extract module name from file path
        module_name = os.path.splitext(os.path.basename(abs_path))[0]
        
        # Import the module
        spec = importlib.util.spec_from_file_location(module_name, abs_path)
        if spec is None:
            print(f"Error: Could not import {module_path}")
            return False
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        # Run the specified function
        if hasattr(module, function_name):
            func = getattr(module, function_name)
            func()
            return True
        else:
            print(f"Error: Function {function_name} not found in module {module_name}")
            return False
    except Exception as e:
        print(f"Error running {module_path}: {str(e)}")
        return False

def main():
    """Run all tests for the backend"""
    print("Starting backend tests...")
    
    # Store current directory
    start_dir = os.getcwd()
    
    # Change to backend directory if needed
    if os.path.basename(start_dir) != 'Backend':
        backend_dir = os.path.join(start_dir, 'Backend')
        if os.path.exists(backend_dir):
            os.chdir(backend_dir)
            print(f"Changed to directory: {backend_dir}")
        else:
            print(f"Backend directory not found at {backend_dir}. Using current directory.")
    
    # List of tests to run
    tests = [
        # Test validation of test cases
        {"type": "import", "path": "validate_test_cases.py", "function": "validate_test_cases"},
        
        # Test mock knowledge graph (doesn't require Neo4j)
        {"type": "import", "path": "test_mock_kg.py", "function": "run_tests"},
        
        # Test knowledge graph with actual Neo4j (skip if not available)
        {"type": "command", "command": "python -m unittest test_kg.py"}
    ]
    
    # Run each test
    success = True
    for test in tests:
        if test["type"] == "command":
            result = run_command(test["command"], test.get("cwd"))
            if result != 0:
                print(f"Command failed with exit code {result}")
                success = False
        elif test["type"] == "import":
            result = import_and_run(test["path"], test["function"])
            if not result:
                print(f"Function {test['function']} failed")
                success = False
    
    # Return to original directory
    os.chdir(start_dir)
    
    if success:
        print("\n✅ All tests passed! The backend is working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Please fix the issues before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 