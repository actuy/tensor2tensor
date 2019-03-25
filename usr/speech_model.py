from tensor2tensor.models.transformer import transformer_base
from tensor2tensor.utils import registry


@registry.register_hparams()
def transformer_ljspeech():
    hparams = transformer_base()
    # set_ljspeech_hparams(hparams)
    return hparams
