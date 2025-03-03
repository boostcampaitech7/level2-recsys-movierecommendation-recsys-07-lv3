import torch
import numpy as np
from tqdm import tqdm

from ..trainer import Trainer

class RecVAETrainer(Trainer):

    def train(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        user_info = torch.tensor(self.dataset.user_item_matrix.todense()).to(device)
        dataset = torch.utils.data.TensorDataset(user_info)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
        )
        self.model.to(device)

        encoder_params = set(self.model.encoder.parameters())
        decoder_params = set(self.model.decoder.parameters())
        optimizer_encoder = torch.optim.Adam(encoder_params, lr=self.config['learning_rate'])
        optimizer_decoder = torch.optim.Adam(decoder_params, lr=self.config['learning_rate'])

        def run_optimizer(cur_optimizer, cur_epoch, dropout):
            for _ in range(cur_epoch):
                for (batch_user_info, ) in dataloader:
                    cur_optimizer.zero_grad()
                    batch_user_info = batch_user_info.to(device).to(torch.float32)
                    _, loss = self.model(batch_user_info, dropout_rate=dropout)
                    loss.backward()
                    cur_optimizer.step()
        
        for epoch in range(self.config['epoch_num']):
            run_optimizer(optimizer_encoder, 3, 0.5)
            self.model.update_prior()
            run_optimizer(optimizer_decoder, 1, 0)
            print(f'epoch: {epoch}')
            if epoch % self.config['validate_interval'] == 0:
                self.validate()

