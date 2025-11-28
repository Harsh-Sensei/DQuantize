import re

def extract_final_answer_from_output(text: str) -> str:
    """
    Extract the final numeric answer from a chain-of-thought solution.
    Returns the number as a string. If none found, returns the full text.
    """
    # Look for common phrases followed by the answer
    patterns = [
        r"final result is\s*([-\d]+)",   # "The final result is 72"
        r"conclusively[:\s]*([-\d]+)",   # "Conclusively: 10"
        r"therefore\s*[^\d]*([-\d]+)",   # "Therefore, Betty now has ... 95"
        r"= *([-\d]+)"                    # Matches = 72, = 5, etc.
    ]
    
    text = text.lower()  # make matching case-insensitive
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            return match.group(1)
    
    # Fallback: return the first number in text
    match = re.search(r"[-+]?\d+", text)
    if match:
        return match.group(0)
    
    # fallback: return full text if nothing found
    return text.strip()


def show_accuracy(outputs, preds):
    assert len(outputs) == len(preds), "Number of outputs and predictions must match."
    correct = 0
    total = len(outputs)
    for output, truth in zip(outputs, preds):
        if output == truth:
            correct += 1
    accuracy = correct / total if total > 0 else 0.0
    print(f"GSM8K Exact Match Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    return accuracy