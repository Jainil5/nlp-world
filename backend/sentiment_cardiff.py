from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
from process_text import filter_text

import_model = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
tokenizer = AutoTokenizer.from_pretrained(import_model)
model = AutoModelForSequenceClassification.from_pretrained(import_model)


def get_sentiment(input_text):
    text = filter_text(input_text)
    token = tokenizer.encode(text, return_tensors="pt")
    result = model(token)
    score = int(torch.argmax(result.logits))
    sentiment = ["negative", "neutral", "positive"]
    return sentiment[score]



text1= f"""
The Hubble Space Telescope has spotted water vapor erupting from the icy surface of Jupiter's moon Europa. 
The discovery strengthens the possibility of finding life elsewhere in our solar system, scientists say.
Europa has long been considered one of the top places to search for signs of life. 
It has a huge global ocean beneath its frozen crust. This ocean is about 100 kilometers deep and contains twice as much water as all of Earth's oceans combined.
"""

text2 = f"I am so angry rn you cant believe."


# print(get_sentiment(text1))
