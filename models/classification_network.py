import torch.nn

class ClassificationNetwork(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self,x):
        print(x.shape)
        return x.view(x.shape[0],-1).mean(dim=-1)