import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset

# Use the prediction entropy as uncertainty
class EntropySampling():
    def __init__(self, net, label_dataset=None, unlabel_dataset=None,):
        self.label_dataset = label_dataset
        self.unlabel_dataset = unlabel_dataset
        self.net = net
    
    def predict_prob(self, data, device='cuda'):
        self.net.to(device)
        self.net.eval()
        probs = torch.zeros([len(data), 2, 448, 448])
        x_names = [None] * len(data)
        loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=1)
        with torch.no_grad():
            # for x, y, idxs in loader:
            for i_batch, sampled_batch in enumerate(loader):
                volume_valbatch, label_valbatch = sampled_batch['image'], sampled_batch['label']
                x, y = volume_valbatch.cuda(), label_valbatch.cuda()
                prob, _ = self.net(x) # torch.Size([8, 2, 448, 448])
                probs[i_batch] = prob.cpu() 
                x_names[i_batch] = sampled_batch['name'][0]
        probs = F.softmax(probs, dim=1)  # Convert logits to probabilities
        return probs, np.array(x_names)

    def query(self, n):
        unlabeled_data = self.unlabel_dataset
        unlabeled_idxs = np.array(range(len(self.unlabel_dataset)))
        probs, sample_names = self.predict_prob(unlabeled_data) #([N, 1, 448, 448])
        # log_probs = torch.log(probs)#([N, 1, 448, 448])
        log_probs = F.log_softmax(probs, dim=1)
        uncertainties = (probs*log_probs).sum((1,2,3))#([N])
        selected_idx = unlabeled_idxs[uncertainties.sort()[1][:n].cpu().numpy()]
        return sample_names[selected_idx]
    