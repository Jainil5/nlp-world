import re

def filter_text(input_text):
    cleaned_text = re.sub(r'\s+', ' ', input_text)
    cleaned_text = re.sub(r"""[^a-zA-Z0-9\s.!,"']""", '', cleaned_text)
    return cleaned_text