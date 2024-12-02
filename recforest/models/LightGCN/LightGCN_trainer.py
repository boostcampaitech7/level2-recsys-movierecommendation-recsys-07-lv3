import torch
from tqdm import tqdm

from ...utils.sampler import NegativeSampler
from ...utils.loss import bpr_loss
from ..trainer import Trainer

class LightGCNTrainer(Trainer):
        
    def train(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        sampler = NegativeSampler(self.dataset, self.config['sample_num_per_user'], 1)

        self.model.to(device)
        for epoch in range(self.config['epoch_num']):
            total_loss = 0
            pairwise_samples = sampler.get_samples().to(device)
            dataset = torch.utils.data.TensorDataset(*pairwise_samples.T)
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
            )
            for users, pos_samples, *neg_samples_list in tqdm(dataloader):
                optimizer.zero_grad()
                pos_scores = self.model(users, pos_samples)
                neg_scores_list = []
                for neg_samples in neg_samples_list:
                    neg_scores_list.append(self.model(users, neg_samples))
                loss = bpr_loss(pos_scores, *neg_scores_list)
                loss.backward()
                optimizer.step() 
                total_loss += loss.item()
                
            print(f'avg_loss: {total_loss / len(dataloader)}')
            if (self.dataset.test_interactions is not None and 
                epoch % self.config['validate_interval'] == 0):
                self.validate()
