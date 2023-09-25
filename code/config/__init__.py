from .TestConfig import get_parser_test
from .TrainConfig import get_parser_train

def get_parser(split='train'):
    if split == 'train':
        return get_parser_train()
    else:
        return get_parser_test()