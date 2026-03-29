"""
DistilBERT text classifier wrapper.
"""

from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast


def get_model(num_labels: int = 4):
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels,
    )
    return model


def get_tokenizer():
    return DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
