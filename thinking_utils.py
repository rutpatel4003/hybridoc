import re
from typing import Optional

# Regex patterns for thinking tags (handles multiline and nested cases)
THINK_PATTERN = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
INCOMPLETE_THINK_OPEN = re.compile(r'<think>.*$', re.DOTALL | re.IGNORECASE)


def strip_thinking_tags(text: str, aggressive: bool = True) -> str:
    """
    Strip <think>...</think> blocks from text
    """
    if not text:
        return text

    # Remove complete thinking blocks
    cleaned = THINK_PATTERN.sub('', text)

    # Remove incomplete thinking blocks (if model output was truncated)
    if aggressive and '<think>' in cleaned.lower():
        cleaned = INCOMPLETE_THINK_OPEN.sub('', cleaned)

    # Clean up excessive whitespace
    cleaned = cleaned.strip()
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Max 2 consecutive newlines

    return cleaned


def extract_thinking_content(text: str) -> Optional[str]:
    """
    Extract ONLY the thinking process (for debugging/analysis).
    """
    match = THINK_PATTERN.search(text)
    if match:
        content = match.group(0)
        # Remove tags
        content = content.replace('<think>', '').replace('</think>', '')
        return content.strip()
    return None


def has_thinking_tags(text: str) -> bool:
    """Check if text contains thinking tags."""
    return '<think>' in text.lower()


# Example usage and tests
if __name__ == "__main__":
    test_cases = [
        ("<think>Let me analyze this...</think>The answer is 42", "The answer is 42"),
        ("<think>Step 1: parse\nStep 2: compute</think>\n\nResult: 100", "Result: 100"),
        ("<think>incomplete", ""),  # Aggressive mode
        ("No thinking tags here", "No thinking tags here"),
        ("<think>nested <think>blocks</think></think>Answer", "Answer"),
    ]

    print("Testing strip_thinking_tags:")
    for input_text, expected in test_cases:
        result = strip_thinking_tags(input_text, aggressive=True)
        status = "CORRECT" if result.strip() == expected.strip() else "WRONG"
        print(f"{status} Input: {input_text[:50]}...")
        print(f"  Output: {result}")
        print()
