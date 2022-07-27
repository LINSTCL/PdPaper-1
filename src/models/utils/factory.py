import logging

logger = logging.getLogger(__name__)

from ..tresnet import TResnetM


def create_model(args):
    """Create a model
    """
    model_params = {'args': args, 'num_classes': args.num_classes}
    args = model_params['args']

    model = TResnetM(model_params)

    return model
