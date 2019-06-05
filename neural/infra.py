import torch
import numpy as np
from torchvision.utils import save_image

from time import time

import sys
sys.path.append('../pytorch-ssim')
import pytorch_ssim

def train(model, optimizer, loss_fn, train_loader, device, epoch):
    model.train()
    epoch_loss = 0

    num_updates = 0
    start_time = time()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        recon_batch, mu, logvar = model.forward(data)
        loss = loss_fn(recon_batch, data, mu, logvar)
        loss.backward()
        epoch_loss += loss.item()

        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)
            )
        )
        num_updates += 1

    total_time = time() - start_time
    print('==> PERFORMED UPDATES @ ' + str(num_updates / total_time) + ' updates/second.')
    print('==> EPOCH COMPLETED AFTER ' + str(total_time) + ' seconds.')

    average_loss = epoch_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, average_loss))

    return average_loss


def test(model, loss_fn, test_loader, device, epoch):
    model.eval()
    test_loss = 0

    batch_ssims = []
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model.forward(data)
            test_loss += loss_fn(recon_batch, data, mu, logvar).item()

            curr_batch_ssim = pytorch_ssim.ssim(recon_batch, data)
            batch_ssims.append(curr_batch_ssim.item())

            if i == 0:
                comparison = torch.cat([data[:20, :, :, :], recon_batch[:20, :, :, :]])
                save_image(comparison, 'results/recon_' + str(epoch) + '.png', nrow = 20)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

    batch_ssim = np.array(batch_ssims).mean()
    print('=========> AVERAGE SSIM: {:.4f}'.format(batch_ssim))

    return test_loss, batch_ssim
