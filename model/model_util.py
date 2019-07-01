import torch


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

def save_model(state, file_name):
    torch.save(state, file_name)

def load_model(model, file_name):
    state_dict = torch.load(file_name)
    model.load_state_dict(state_dict)
