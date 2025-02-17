# Technical Report: SWE-bench Local Testing with Ollama

## 1. Project Overview
This project aims to evaluate the performance of locally hosted LLMs (specifically llama3 via Ollama) on software engineering tasks using the SWE-bench framework. The goal is to assess the model's ability to generate accurate code fixes for real-world programming issues.

## 2. Implementation Details

### 2.1 Architecture
- **Model Integration**: Implemented OllamaModel class for API interaction with locally hosted Ollama server
- **Test Case Processing**: Custom JSON-based test case format for initial testing
- **Evaluation Logic**: Basic similarity scoring and exact match metrics

### 2.2 Key Components
```python
class OllamaModel:
    # Handles API interactions with Ollama server
    # Manages prompt construction and response processing
```

### 2.3 Test Case Format
Initially implemented a simplified test case format:
```json
{
    "test_case_id": "string",
    "problem_description": "string",
    "base_code": "string",
    "target_file": "string",
    "target_line": "integer",
    "correct_patch": "string"
}
```

## 3. Research Findings

### 3.1 SWE-bench Official Format
Through investigation of the official SWE-bench repository, we discovered their actual format:
- **Input**: Problem statement + codebase (repo + base_commit)
- **Output**: Diff patch in specific format
- **Evaluation**: Test-based verification using repository's test suite

### 3.2 Available Datasets
SWE-bench provides multiple dataset splits:
- SWE-bench_Lite: 300 test instances
- SWE-bench_Verified: 500 test instances
- SWE-bench: 225 dev + 2294 test instances
- SWE-bench_Multimodal: 102 dev instances

### 3.3 Initial Results
Our evaluation of the Ollama-hosted llama3 model across five diverse test cases revealed varying levels of performance in code generation and bug fixing capabilities. The model demonstrated significant variance in its ability to generate correct solutions, with similarity scores ranging from 4.76% to 58.82%.

The transformers library test case yielded the highest performance, achieving a 58.82% similarity score. The model successfully implemented a new approach to token handling, though it deviated from the expected solution by introducing a helper method for padding token detection. This suggests the model's strength in handling string manipulation and token processing tasks.

The requests library security fix presented a moderate success with a 42.86% similarity score. While the model correctly identified the need for additional verification checks, it missed crucial security considerations such as the CURL_CA_BUNDLE environment variable, highlighting a potential gap in security-focused code generation.

The pandas datetime handling case achieved a 30% similarity score. The model attempted to address timezone awareness but introduced syntax errors and failed to properly handle NA values, indicating challenges with complex data type manipulations.

The FastAPI dependency injection fix scored 18.92%, revealing significant difficulties with async/await patterns and complex state management. Though the model maintained the basic async structure, it missed critical error handling patterns and produced an incomplete solution for handling default values.

Surprisingly, the PyTorch tensor manipulation case, despite its relatively straightforward nature, received the lowest score of 4.76%. While the model's solution was more concise, it deviated significantly from the expected implementation, particularly in handling empty tensors.

These results highlight several key insights about the model's capabilities:

1. The model performs best when dealing with straightforward input/output transformations and string manipulation tasks, as evidenced by the transformers test case.

2. Complex state management, particularly in async contexts, presents a significant challenge, as shown by the FastAPI example.

3. Security-critical modifications require additional guidance or constraints to ensure all security considerations are properly addressed.

4. The model occasionally opts for creative but potentially problematic solutions, as seen in the PyTorch case, where a more concise approach led to reduced accuracy.

5. Code generation quality remains inconsistent across different domains and complexity levels, suggesting the need for domain-specific fine-tuning or improved prompting strategies.

## 4. Challenges and Limitations

### 4.1 Current Implementation
1. Test cases don't match official SWE-bench format
2. Evaluation metrics differ from official framework
3. Limited test suite compared to full SWE-bench dataset

### 4.2 Technical Constraints
1. Local resource limitations for large-scale testing
2. Potential performance impact of using Ollama API vs direct model integration

## 5. Future Work

### 5.1 Immediate Tasks
1. Align test case format with official SWE-bench structure
2. Implement proper diff-based evaluation
3. Integrate with official test suites

### 5.2 Long-term Goals
1. Expand test coverage using official SWE-bench datasets
2. Implement more sophisticated evaluation metrics
3. Compare performance across different model versions

## 6. Conclusions
The initial implementation provides a foundation for local LLM testing but requires significant alignment with official SWE-bench standards. Early results show promise in the model's ability to generate code fixes, but more rigorous evaluation is needed using official benchmarks.

## 7. References
1. SWE-bench Repository: https://github.com/swe-bench/SWE-bench
2. SWE-bench Experiments: https://github.com/swe-bench/experiments
3. Ollama Documentation: https://ollama.ai/docs
