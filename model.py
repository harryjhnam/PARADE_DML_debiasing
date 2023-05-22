import torch
import torch.nn as nn

from gradient_reversal import GradientReversal


class PARADE(nn.Module):

    def __init__(self, input_dim, filter_dim, grad_reverse=0.2):
        super(PARADE, self).__init__()

        self.f_target = nn.Linear(input_dim, filter_dim)
        self.f_SA = nn.Linear(input_dim, filter_dim)

        self.grad_reverse_layer = GradientReversal(alpha=grad_reverse)

        self.c_SA = nn.Linear(filter_dim, filter_dim)

    def forward(self, CLIP_image_feature):
        
        E_target = self.f_target(CLIP_image_feature) # (batch_size, filter_dim)
        E_SA = self.f_SA(CLIP_image_feature) # (batch_size, filter_dim)
        
        rev_E_target = self.grad_reverse_layer(E_target) # (batch_size, filter_dim)
        rev_E_SA = self.grad_reverse_layer(E_SA) # (batch_size, filter_dim)
        
        Corr = torch.norm( rev_E_target * self.c_SA(rev_E_SA), p=2, dim=-1 ) # (batch_size)
        
        return E_target, E_SA, Corr
    

if __name__ == '__main__':

    parade = PARADE(512, 256)
    dummy_input = torch.Tensor(1024, 512)
    E_target, E_SA, Corr = parade(dummy_input)
    
    print(f"- input size: {dummy_input.size()}")
    print(f"- outputs of the PARADE model are:")
    print(f"    Target embedding: {E_target.size()}")
    print(f"    Sensitive Attribute embedding: {E_SA.size()}")
    print(f"    Correlation term: {Corr.size()}")
