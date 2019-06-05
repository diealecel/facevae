import torch
import numpy as np
from torch import optim
from os.path import join

from neural.infra import train, test
from neural.utils import get_data_loader
from neural.loss import bce_kld_loss

# Models
from models import BaselineVAE
from models import VAE1
from models import VAE2
from models import VAE3
from models import VAE4
from models import VAE5
from models import VAE6_1
from models import VAE6_10
from models import VAE6_100
from models import VAE7

from argparse import ArgumentParser

TRAIN_LOC = 'dataset/10k_500_split/train'
TEST_LOC = 'dataset/10k_500_split/test'
RESULTS_DIR = 'results'
MODEL_SAVE_DIR = 'models/saved'
MODEL_SAVE_EXT = '.model'

def save_results(train_loss, test_loss, ssim):
    train_loss, test_loss, ssim = np.array(train_loss), np.array(test_loss), np.array(ssim)
    np.save(join(RESULTS_DIR, 'train_loss.npy'), train_loss)
    np.save(join(RESULTS_DIR, 'test_loss.npy'), test_loss)
    np.save(join(RESULTS_DIR, 'ssim.npy'), ssim)


def main(args, device):
    train_loader = get_data_loader(TRAIN_LOC, args.batch_size)
    test_loader = get_data_loader(TEST_LOC, args.batch_size, shuffle = False)

    # Construct model and optimizer.
    model = VAE7()
    if torch.cuda.device_count() > 1:
        print('Let\'s burn ' + str(torch.cuda.device_count()) + 'x the amount of money!')
        model = torch.nn.DataParallel(model)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay = 1e-2)

    train_loss = []
    test_loss = []
    ssim = []
    for epoch in range(1, args.epochs + 1):
        curr_train_loss = train(model, optimizer, bce_kld_loss, train_loader, device, epoch)
        curr_test_loss, curr_ssim = test(model, bce_kld_loss, test_loader, device, epoch)

        train_loss.append(curr_train_loss)
        test_loss.append(curr_test_loss)
        ssim.append(curr_ssim)

    save_results(train_loss, test_loss, ssim)
    if args.save is not None:
        model_filename = args.save + MODEL_SAVE_EXT
        model_path = join(MODEL_SAVE_DIR, model_filename)
        torch.save(model, model_path)
        print('\nModel saved as ' + model_filename + '.')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--no-cuda', action = 'store_true',
                        help = 'disables CUDA training')
    parser.add_argument('--save', type = str, default = None,
                        help = 'filename to save model at end of training')
    parser.add_argument('--epochs', type = int, default = 10,
                        help = 'number of epochs to train (default: 10)')
    parser.add_argument('--batch_size', type = int, default = 64,
                        help = 'number of examples per batch (default: 32)')
    parser.add_argument('-lr', '--learning_rate', type = float, default = 1e-3,
                        help = 'constant learning rate to be used (default: 1e-3)')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device('cuda' if args.cuda else 'cpu')
    main(args, device)
