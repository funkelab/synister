import torch
import torch.nn.functional as F
import numpy as np

def get_integrated_gradients(model, checkpoint_file, raw, baseline, target_id, integration_steps=50):
    """
    Given a pytorch model whose output are class logits this
    function returns integrated gradients for any given input and 
    baselinea.

    Arguments:

    model (``object``):
        
        Pytorch model

    checkpoint_file(``string``):

        Path to model checkpoint file

    raw (``ndarray``): 
    
        normalized input volume

    baseline (``ndarray``, same shape as raw)

        baseline to compare against, e.g. np.zeros(np.shape(raw))

    target_id(``int``):

        class id of raw

    integration_steps(``int``, optional):

        number of steps between baseline and raw to consider for integration
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gradients = []
    # Get inputs on line between baseline and raw
    inputs = get_inputs(baseline, raw)
    for in_raw in inputs:
        raw_batched = in_raw.reshape((1,) + np.shape(raw))
        raw_batched_tensor = torch.tensor(raw, device=device, requires_grad=True)
        
        # predict
        output = model(raw=raw_batched_tensor)
        output = F.softmax(output, dim=1)
       
        # get gradient
        index = np.ones((output.size()[0], 1)) * target_idx
        index = torch.tensor(index, dtype=torch.int64, device=device)
        output = output.gather(1, index)
        model.zero_grad()
        output.backward()
        gradient = raw_batched_tensor.grad.detach().cpu().numpy()[0]
        gradients.append(gradient)

    integrated_grads = np.sum(gradients[:-1], axis=0)
    average_grads = np.average(gradients[:-1], axis=0)
    return output, average_grads, integrated_grads, raw

def get_inputs(baseline, raw, steps):
    inputs = [baseline + (float(i)/steps) * (raw - baseline) for i in range(0, steps+1)]
    return inputs
