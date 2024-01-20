from .utils import tokenize, _transform
from .model import MGABase
from .gpt import GPT2LMHeadModel
from .mg_model import MultiGrainModel

def get_model(config):
    if config.resume:
        return globals()[config.model].load_from_checkpoint(config.resume,config=config)   
    else:
        return globals()[config.model](config)