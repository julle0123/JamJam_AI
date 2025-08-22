# app/graph/prompts.py
PROMPT_PATH = {
    "test": "app/prompt/test.txt",
}

def load_prompt_template(name: str) -> str:
    """
    지정된 프롬프트 이름에 해당하는 txt 템플릿 로드
    """
    path = PROMPT_PATH.get(name, PROMPT_PATH["test"])
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
