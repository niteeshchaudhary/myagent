import json
import re


def extract_json(text: str):
    """
    Extract and parse JSON from raw LLM output.
    Handles cases where the LLM returns text around JSON or formats it incorrectly.
    """

    # Try direct JSON first
    try:
        return json.loads(text)
    except:
        pass

    # Find JSON inside ```json ... ``` blocks
    fenced = re.search(r"```json(.*?)```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except:
            pass

    # Attempt to extract first {...} structure
    bracketed = re.search(r"\{.*\}", text, re.DOTALL)
    if bracketed:
        cleaned = bracketed.group(0)
        try:
            return json.loads(cleaned)
        except:
            try:
                return json.loads(fix_common_json_issues(cleaned))
            except:
                pass

    raise ValueError(f"Could not parse JSON from text:\n{text}")


def fix_common_json_issues(text: str) -> str:
    """
    Fix common JSON issues:
    - trailing commas
    - single quotes instead of double quotes
    - missing quotes around keys
    """

    # Replace single quotes with double quotes
    text = text.replace("'", '"')

    # Remove trailing commas
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)

    # Add quotes around unquoted keys: key: "value" → "key": "value"
    text = re.sub(r"(\w+)\s*:", r'"\1":', text)

    return text
