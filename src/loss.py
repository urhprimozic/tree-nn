import torch

# exp(1 - x)




def EMSG(output, model):
    pass

def ROMSG(output, model, dim=None):
    '''
    Relu of One Minus Sizeof Grad (prebere se rom es đi)
    TODO
    Returns sum( ReLU(1 - |grad|) ), 
    '''
    grads = torch.autograd.grad(
    outputs=output,
    inputs=[p for p in model.parameters() if p.requires_grad],
    create_graph=True 
    )

    ans  = 0

    for g in grads:
        norm = torch.norm(g)
        ans += torch.relu(1 - norm) 
    return ans


    
