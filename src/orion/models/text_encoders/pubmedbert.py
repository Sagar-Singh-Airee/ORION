from .registry import TEXT_ENCODERS
from .xlm_roberta import HuggingFaceTextEncoder


@TEXT_ENCODERS.register("pubmedbert")
class PubMedBert(HuggingFaceTextEncoder):
    def __init__(self, **kwargs: object):
        super().__init__("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", **kwargs)
