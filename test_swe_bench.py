import os
import json
import requests
import re
from pathlib import Path
from typing import Dict, Any

class OllamaModel:
    def __init__(self, model_name="llama2"):
        self.model_name = model_name
        self.api_base = "http://localhost:11434"
        
    def generate(self, prompt: str, max_length: int = 2048, temperature: float = 0.7) -> str:
        """Generate text using Ollama API."""
        response = requests.post(
            f"{self.api_base}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_length,
            },
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.text}")
            
        # Ollama streams the response, so we need to concatenate all the chunks
        response_text = ""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                response_text += chunk.get("response", "")
                
                # Check if this is the last chunk
                if chunk.get("done", False):
                    break
        
        return response_text.strip()

def validate_test_case(test_case: Dict[str, Any]) -> bool:
    """Validate that a test case has all required fields."""
    required_fields = [
        "test_case_id",
        "problem_description",
        "base_code",
        "target_file",
        "target_line"
    ]
    return all(field in test_case for field in required_fields)

def construct_prompt(problem_description: str, base_code: str, target_file: str, target_line: int) -> str:
    """Construct a prompt for the model."""
    return f"""You are an expert software engineer. Please help fix the following code:

Problem Description:
{problem_description}

File: {target_file}
Line: {target_line}

Current Code:
```
{base_code}
```

Please provide ONLY the complete fixed version of the code. Do not include any explanations, comments, or markdown formatting. The output should be exactly the fixed code and nothing else.
"""

def process_test_case(model: OllamaModel, test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single test case and return the results."""
    if not validate_test_case(test_case):
        raise ValueError(f"Test case {test_case.get('test_case_id', 'unknown')} is missing required fields")
    
    # Construct the prompt
    prompt = construct_prompt(
        problem_description=test_case["problem_description"],
        base_code=test_case["base_code"],
        target_file=test_case["target_file"],
        target_line=test_case["target_line"]
    )
    
    # Generate solution using Ollama
    solution = model.generate(prompt)
    
    # Return results in the expected format
    return {
        "test_case_id": test_case["test_case_id"],
        "generated_solution": solution,
        "metadata": {
            "model": f"ollama-{model.model_name}",
            "temperature": 0.7,
            "max_length": 2048
        }
    }

def normalize_code(code):
    """Normalize code by removing whitespace, comments, and standardizing syntax."""
    # Remove comments
    code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
    # Remove empty lines
    code = '\n'.join(line.strip() for line in code.split('\n') if line.strip())
    # Standardize whitespace
    code = re.sub(r'\s+', ' ', code)
    return code

def tokenize_code(code):
    """Convert code into tokens for comparison."""
    # Basic tokenization - split on spaces and punctuation
    tokens = re.findall(r'\w+|[^\w\s]', code)
    return [t.lower() for t in tokens]

def get_structure(code):
    structure = []
    for line in code.split('\n'):
        line = line.strip()
        if line.startswith(('def ', 'class ', 'if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except:', 'finally:')):
            structure.append(line)
    return structure

def evaluate_solution(generated, correct):
    """Evaluate the generated solution against the correct solution using multiple metrics."""
    metrics = {}
    
    # Clean up the code
    generated = generated.strip().replace('```', '')
    correct = correct.strip()
    
    # 1. Line-based similarity (original metric)
    generated_lines = set(generated.split('\n'))
    correct_lines = set(correct.split('\n'))
    common_lines = len(generated_lines.intersection(correct_lines))
    total_lines = len(generated_lines.union(correct_lines))
    metrics['line_similarity'] = common_lines / total_lines if total_lines > 0 else 0
    
    # 2. Token-based similarity
    generated_norm = normalize_code(generated)
    correct_norm = normalize_code(correct)
    generated_tokens = set(tokenize_code(generated_norm))
    correct_tokens = set(tokenize_code(correct_norm))
    common_tokens = len(generated_tokens.intersection(correct_tokens))
    total_tokens = len(generated_tokens.union(correct_tokens))
    metrics['token_similarity'] = common_tokens / total_tokens if total_tokens > 0 else 0
    
    # 3. Structure similarity (based on function definitions, conditionals, etc.)
    generated_struct = get_structure(generated)
    correct_struct = get_structure(correct)
    common_struct = len(set(generated_struct).intersection(set(correct_struct)))
    total_struct = len(set(generated_struct).union(set(correct_struct)))
    metrics['structure_similarity'] = common_struct / total_struct if total_struct > 0 else 0
    
    # 4. Length similarity
    max_len = max(len(generated), len(correct))
    length_diff = abs(len(generated) - len(correct))
    metrics['length_similarity'] = 1 - (length_diff / max_len) if max_len > 0 else 0
    
    # Calculate weighted average similarity
    weights = {
        'line_similarity': 0.3,
        'token_similarity': 0.4,
        'structure_similarity': 0.2,
        'length_similarity': 0.1
    }
    
    final_similarity = sum(metrics[k] * weights[k] for k in weights)
    
    return {
        'exact_match': generated == correct,
        'similarity': final_similarity,
        'metrics': {
            'line_similarity': metrics['line_similarity'],
            'token_similarity': metrics['token_similarity'],
            'structure_similarity': metrics['structure_similarity'],
            'length_similarity': metrics['length_similarity']
        }
    }

def main():
    # First, check if Ollama is running
    try:
        requests.get("http://localhost:11434/api/version")
    except requests.exceptions.ConnectionError:
        print("Error: Ollama is not running. Please start Ollama first using 'ollama serve'")
        return

    # Set paths
    current_dir = Path(__file__).parent
    test_cases_file = current_dir / "test_cases.json"
    
    # Load test cases
    print(f"Loading test cases from {test_cases_file}")
    with open(test_cases_file, 'r') as f:
        data = json.load(f)
        test_cases = data["test_cases"]
    
    # Initialize Ollama model
    print("Initializing Ollama model...")
    model = OllamaModel(model_name="llama3")
    
    # Process test cases
    results = []
    for test_case in test_cases:
        print(f"\nProcessing test case: {test_case['test_case_id']}")
        
        try:
            # Generate solution
            result = process_test_case(model, test_case)
            
            # Evaluate solution if correct patch is available
            if "correct_patch" in test_case:
                evaluation = evaluate_solution(
                    result["generated_solution"],
                    test_case["correct_patch"]
                )
                result["evaluation"] = evaluation
                
                print("\nEvaluation Results:")
                print(f"Exact Match: {evaluation['exact_match']}")
                print(f"Similarity Score: {evaluation['similarity']:.2%}")
                print("Metrics:")
                for metric, value in evaluation['metrics'].items():
                    print(f"{metric}: {value:.2%}")
            
            results.append(result)
            
            # Print the generated solution
            print("\nGenerated Solution:")
            print(result["generated_solution"])
            
        except Exception as e:
            print(f"Error processing test case {test_case['test_case_id']}: {str(e)}")
            continue
    
    # Save results
    output_file = current_dir / "results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
