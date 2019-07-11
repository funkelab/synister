import torch.nn.functional as F
import torch
import Vgg3D
import logging

logger = logging.getLogger(__name__)

def predict(raw_batched,
            model):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw_batched_tensor = torch.tensor(raw_batched, device=device)
    output = model(raw=raw_batched_tensor)
    output = F.softmax(output, dim=1)
    return output


def init_vgg(checkpoint_file,
             input_shape,
             fmaps=32):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Vgg3D.Vgg3D(input_size=input_shape, fmaps=fmaps)
    model.to(device)
    logger.info("Init vgg with checkpoint: ", checkpoint_file)
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
