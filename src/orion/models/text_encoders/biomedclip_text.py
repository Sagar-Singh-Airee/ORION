"""BiomedCLIP text branch is currently represented by its public HF encoder."""
from .registry import TEXT_ENCODERS
from .xlm_roberta import HuggingFaceTextEncoder


@TEXT_ENCODERS.register("biomedclip_text")
class BiomedClipText(HuggingFaceTextEncoder):
    def __init__(self, **kwargs: object):
        super().__init__("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", **kwargs)
