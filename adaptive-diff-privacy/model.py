"""
DistilBERT text classifier wrapper.
"""

import logging
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

# Suppress expected warnings about discarded MLM head / new classifier head
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


def get_model(num_labels: int = 4):
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    return model


def get_tokenizer():
    return DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
