import re

def clean_text_for_speech(text: str) -> str:
    """
    Cleans raw markdown or text symbols that sound bad when spoken.
    """
    # Remove markdown image tags: ![alt](url)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # Remove standard markdown links: [text](url) -> text
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    # Remove formatting characters like *, _, #, `, etc.
    text = re.sub(r"[*_#`~-]", " ", text)
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text
