from .registry import TEXT_ENCODERS
from .xlm_roberta import HuggingFaceTextEncoder


@TEXT_ENCODERS.register("bert_base_uncased")
class BertBase(HuggingFaceTextEncoder):
    def __init__(self, **kwargs: object):
        super().__init__("bert-base-uncased", **kwargs)
