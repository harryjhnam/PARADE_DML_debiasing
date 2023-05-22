import torch
import torch.nn as nn

class PARADE_loss():
    
    def __init__(self, alpha=0.1, return_losses = False):
        self.triplet_margin_loss = nn.TripletMarginWithDistanceLoss(
            distance_function = nn.CosineSimilarity(dim=-1, eps=1e-6),
            reduction = "mean"
            )
        self.alpha = alpha
        self.return_losses = return_losses
    
    def loss(self, anchor_embeds, pos_embeds, neg_embeds, correlations):
        # anchor_embeds = anchor_target_embeds, anchor_sa_embeds (each size: [batch_size, embed_dim])
        # pos_embeds = pos_target_embeds, pos_sa_embeds (each size: [batch_size, embed_dim])
        # neg_embeds = neg_target_embeds, neg_sa_embeds (each size: [batch_size, embed_dim])
        # correlations = [batch_size]    
        
        self.target_loss = self.triplet_margin_loss(anchor_embeds[0], pos_embeds[0], neg_embeds[0])
        self.sa_loss = self.triplet_margin_loss(anchor_embeds[1], pos_embeds[1], neg_embeds[1])
        self.corr_term = correlations.mean() if len(correlations) != 0 else None

        total_loss = self.target_loss + self.alpha * self.sa_loss if self.corr_term is None\
                    else self.target_loss + self.alpha * self.sa_loss - self.corr_term

        if self.return_losses:
            if self.corr_term is not None:
                return total_loss, self.target_loss.item(), self.sa_loss.item(), self.corr_term.item()
            else:
                return total_loss, self.target_loss.item(), self.sa_loss.item(), 0.0
        
        return total_loss


if __name__ == "__main__":

    loss_fn = PARADE_loss(alpha=0.1, return_losses=True).loss

    # dummy inputs
    t_anc, t_pos, t_neg = torch.Tensor(1024, 128), torch.Tensor(1024, 128), torch.Tensor(1024, 128)
    sa_anc, sa_pos, sa_neg = torch.Tensor(1024, 128), torch.Tensor(1024, 128), torch.Tensor(1024, 128)
    Corr = torch.Tensor(1024)

    # loss
    total_loss, target_loss_term, sa_loss_term, corr_term = loss_fn((t_anc, sa_anc), (t_pos, sa_pos), (t_neg, sa_neg), Corr)

    print(f"- Total loss: {total_loss}")
    print(f"    target loss term: {target_loss_term}")
    print(f"    sensitive attribute loss term: {sa_loss_term}")
    print(f"    correlation term: {corr_term}")
