import pytest
from query_data import query_rag
from langchain_community.llms.ollama import Ollama

LLM = "deepseek-r1:7b"

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""

def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = Ollama(model=LLM)
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print("Evaluation prompt:\n", prompt)

    if "true" in evaluation_results_str_cleaned:
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError("Invalid evaluation result. Cannot determine if 'true' or 'false'.")

@pytest.mark.parametrize("question,expected_response", [
    ("What is 1+1? (Answer with the number only)", "2"),
    ("What color paint do you get if you mix yellow and blue paint? (Answer with one word only)", "green")
])
def test_query(question, expected_response):
    assert query_and_validate(question, expected_response)

if __name__ == "__main__":
    pytest.main()
