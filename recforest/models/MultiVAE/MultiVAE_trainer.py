import torch

from ..trainer import Trainer
from ...utils.loss import vae_reg_loss, vae_bce_loss

class MultiVAETrainer(Trainer):

    def train(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        user_info = torch.tensor(self.dataset.user_item_matrix.todense()).to(device)
        dataset = torch.utils.data.TensorDataset(user_info)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
        )
        self.model.to(device)

        for epoch in range(self.config['epoch_num']):
            for (batch_user_info, ) in dataloader:
                optimizer.zero_grad()
                batch_user_info = batch_user_info.to(device).to(torch.float32)
                
                recon_users, mu, log_var = self.model(batch_user_info)
                reg_loss = vae_reg_loss(mu, log_var)
                bce_loss = vae_bce_loss(batch_user_info, recon_users)
                loss = (bce_loss + self.config['beta'] * reg_loss).mean()
                loss.backward()
                optimizer.step()
            
            print(f'epoch: {epoch}')
            if epoch % self.config['validate_interval'] == 0:
                self.validate()

