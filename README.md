# SWE-bench Local LLM Testing

This project provides a framework for testing local LLM models using the [SWE-bench](https://github.com/swe-bench/SWE-bench) benchmark. It allows you to evaluate how well your local language model performs on real-world software engineering tasks.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Update the following paths in `test_swe_bench.py`:
   - `test_cases_file`: Path to your SWE-bench test cases JSON file
   - `model_path`: Path to your local LLM model

## Usage

1. Prepare your test cases in JSON format following the SWE-bench format:
```json
[
  {
    "test_case_id": "unique_id",
    "problem_description": "Description of the problem",
    "base_code": "Original code",
    "target_file": "Path to the file to modify",
    "target_line": "Line number to modify"
  }
]
```

2. Run the test script:
```bash
python test_swe_bench.py
```

The script will:
1. Load your local LLM model
2. Process each test case using SWE-bench's prompting module
3. Generate solutions using your model
4. Run the test cases using SWE-bench's test runner
5. Save the results to `results.json`

## Output

The script generates a `results.json` file containing:
- Generated solutions for each test case
- Test results (pass/fail)
- Metadata about the model and generation parameters

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- Sufficient RAM to load your local LLM model
- SWE-bench test cases in JSON format
