import torch
import numpy as np

import sys
sys.path.append('../pytorch-ssim')

import pytorch_ssim

def train(model, optimizer, loss_fn, train_loader, device, epoch):
    model.train()
    epoch_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)
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
            recon_batch, mu, logvar = model(data)
            test_loss += loss_fn(recon_batch, data, mu, logvar).item()

            shaped_recon = recon_batch.view(data.shape)
            curr_batch_ssim = pytorch_ssim.ssim(shaped_recon, data)
            batch_ssims.append(curr_batch_ssim)

                # comparison = torch.cat([data, recon_batch.view(data.shape[0], 3, 100, 100)])
                # save_image(comparison.cpu(), 'results/recon_' + str(epoch) + '.png', nrow = data.shape[0])

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

    batch_ssim = np.array(batch_ssims).mean()
    print('=========> AVERAGE SSIM: {:.4f}'.format(batch_ssim))

    return test_loss, batch_ssim
