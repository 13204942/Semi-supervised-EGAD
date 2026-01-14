import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset

# The parameter space is obtained by Bayesian model (MC dropout), making the uncertainty of the parameter space as small as possible
# bayesian_active_learning_disagreement_dropout (BALDD)
class HybridSampling():
    def __init__(self, net, label_dataset=None, unlabel_dataset=None, n_drop=10):
        self.label_dataset = label_dataset
        self.unlabel_dataset = unlabel_dataset
        self.net = net
        self.coef = 0.5
        self.n_drop = n_drop

    def normalize_scores(self, scores):
        return (scores - scores.min()) / (scores.max() - scores.min())
    
    def mutual_info(self, img1, img2):
        hgram = np.histogram2d(img1.ravel(), img2.ravel(), 20)[0]
        pxy = hgram / float(np.sum(hgram))
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        px_py = px[:, None] * py[None, :]
        nzs = pxy > 0
        return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

    def calculate_similarity(self, unlabeled_data, labeled_data): # (N_ul, 3, 448, 448) (N_l, 3, 448, 448)
        # Calculate cosine similarity between unlabeled and labeled samples
        simi_mat = cosine_similarity(unlabeled_data.reshape(unlabeled_data.shape[0], -1), 
                                     labeled_data.reshape(labeled_data.shape[0], -1))
        
        simi_mat = simi_mat.mean(axis=1)
        return simi_mat
    
    def calculate_cosine_mi_info(self, unlabeled_dataset, labeled_dataset):
        mi_scores = np.zeros(len(unlabeled_dataset))
        simi_scores = np.zeros(len(unlabeled_dataset))

        for idx, unlabeled_sample in enumerate(unlabeled_dataset):
            mi = np.zeros(len(labeled_dataset))
            simi = np.zeros(len(labeled_dataset))
            unlabeled_img = unlabeled_sample['image'].unsqueeze(0)

            for i, labeled_sample in enumerate(labeled_dataset):
                labeled_img = labeled_sample['image'].unsqueeze(0)
                mi[i] = self.mutual_info(unlabeled_img, labeled_img)
                simi[i] = self.calculate_similarity(unlabeled_img, labeled_img)

            mi_scores[idx] = np.mean(mi)  # loop each labeled sample and get the mutual information
            simi_scores[idx] = np.mean(simi)  # loop each labeled sample and get the mutual information
        return simi_scores, mi_scores
    
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
        return probs, np.array(x_names)    

    def calculate_uncertainty(self, data):
        # unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data(index = index)
        # unlabeled_data = self.unlabel_dataset
        probs, sample_names = self.predict_prob(data)  # ([N, 2, 448, 448])
        # probs = F.softmax(probs, dim=1)  # Convert logits to probabilities
        log_probs = F.log_softmax(probs, dim=1)
        uncertainties = (probs*log_probs).sum((1,2,3))  #([N])
        return uncertainties, sample_names

    def query(self, n):
        # unlabeled_idxs, unlabeled_data, labeled_data = self.dataset.get_data()
        data_len = len(self.unlabel_dataset)
        labeled_data = self.label_dataset
        unlabeled_data = self.unlabel_dataset
        unlabeled_idxs = np.array(range(len(unlabeled_data)))

        # Step 1: Select a subset based on uncertainty
        uncertainty_scores, sample_names = self.calculate_uncertainty(unlabeled_data)
        print(f'uncertainty_scores shape: {uncertainty_scores.shape}')
        print(f'data_len:{data_len}')
        print(f'uncertainty_scores min: {uncertainty_scores.min()}, max: {uncertainty_scores.max()}')
        # You might want to determine a threshold or a subset size for the uncertain samples
        sort_idx = uncertainty_scores.sort(descending=False)[1][:int(data_len/3)]
        # print(sort_idx)
        # uncertain_idxs = unlabeled_idxs[uncertainty_scores.sort()[1][:int(data_len/3)]]
        # uncertain_data = unlabeled_data[uncertainty_scores.sort()[1][:int(data_len/3)]]
        uncertain_idxs = unlabeled_idxs[sort_idx]
        uncertain_data = Subset(unlabeled_data, sort_idx)

        # test_loader = DataLoader(uncertain_data, batch_size=1, shuffle=False, num_workers=2)
        # for i_batch, sampled_batch in enumerate(test_loader):
        #     volume_batch, label_batch, img_name = sampled_batch['image'], sampled_batch['label'], sampled_batch['name']
        #     print(img_name)

        # Step 2: Select the final samples based on diversity within the uncertain subset
        # simi_scores = self.calculate_similarity(uncertain_data, labeled_data)
        # mi_scores = self.calculate_mutual_info(uncertain_data, labeled_data)
        simi_scores, mi_scores = self.calculate_cosine_mi_info(uncertain_data, labeled_data)
        norm_simi_scores = self.normalize_scores(simi_scores)
        norm_mi_scores = self.normalize_scores(mi_scores)

        combined_scores = 2 - self.coef * norm_simi_scores - self.coef * norm_mi_scores
        selected_uncertain_idxs = np.argsort(-combined_scores)[:n]

        # Map back to the original indices in the full unlabeled set
        final_selected_idxs = uncertain_idxs[selected_uncertain_idxs]
        
        return sample_names[final_selected_idxs]
        # return final_selected_idxs
    