from .resnet import ResNet


def create_model(model_cfg):
    name = model_cfg['name']
    args = model_cfg['args']

    if name == 'resnet':
        model = ResNet(**args)
    else:
        raise NotImplementedError(f'Model "{name}" is not supported.')

    return model
