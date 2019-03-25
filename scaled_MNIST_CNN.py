from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset
from torch.utils.data.dataset import Subset
from torchvision import transforms, utils
import sys
import numpy as np

import matplotlib as matplotlib
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    #print(len(train_loader.dataset))
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # print(target)
        # grid = utils.make_grid(data)
        # plt.imshow(grid.numpy().transpose((1, 2, 0)))
        # plt.show()
        # sys.exit()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     print(scale)
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))
            #transforms.ToPILImage()(data[63]).show()
def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # print(target[0:64])
            # grid = utils.make_grid(data[0:64])
            # plt.imshow(grid.numpy().transpose((1, 2, 0)))
            # plt.show()
            # sys.exit()
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    #transforms.ToPILImage()(data[9999]).show()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    d1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    d1.train_labels[:] = 1
    d2 = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize(24),
                           transforms.Pad(padding=2, padding_mode='edge'),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    d2.train_labels[:] = 2
    d3 = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize(22),
                           transforms.Pad(padding=3, padding_mode='edge'),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    d3.train_labels[:] = 3
    d4 = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize(20),
                           transforms.Pad(padding=4, padding_mode='edge'),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    d4.train_labels[:] = 4
    d5 = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize(16),
                           transforms.Pad(padding=6, padding_mode='edge'),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    d5.train_labels[:] = 5
    d6 = datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([
                           transforms.Resize(14),
                           transforms.Pad(padding=7, padding_mode='edge'),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                        ]))
    d6.train_labels[:] = 6
    Comb_data = _ConcatDataset([d1, d2, d3, d4, d5, d6])
    train_loader_all = torch.utils.data.DataLoader(
        Comb_data, batch_size=args.batch_size, shuffle=True, **kwargs)
    train_loader2 = torch.utils.data.DataLoader(
        d2, batch_size=args.batch_size, shuffle=True, **kwargs)
    train_loader3 = torch.utils.data.DataLoader(
        d3, batch_size=args.batch_size, shuffle=True, **kwargs)
    train_loader4 = torch.utils.data.DataLoader(
        d4, batch_size=args.batch_size, shuffle=True, **kwargs)
    train_loader5 = torch.utils.data.DataLoader(
        d5, batch_size=args.batch_size, shuffle=True, **kwargs)
    train_loader6 = torch.utils.data.DataLoader(
        d6, batch_size=args.batch_size, shuffle=True, **kwargs)

    t1 = datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    t1.test_labels[:] = 1
    t2 = datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.Resize(24),
                           transforms.Pad(padding=2, padding_mode='edge'),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    t2.test_labels[:] = 2
    t3 = datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.Resize(22),
                           transforms.Pad(padding=3, padding_mode='edge'),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    t3.test_labels[:] = 3
    t4 = datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.Resize(20),
                           transforms.Pad(padding=4, padding_mode='edge'),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    t4.test_labels[:] = 4
    t5 = datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.Resize(16),
                           transforms.Pad(padding=6, padding_mode='edge'),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    t5.test_labels[:] = 5
    t6 = datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.Resize(14),
                           transforms.Pad(padding=7, padding_mode='edge'),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    t6.test_labels[:] = 6
    comb_test = _ConcatDataset([t1, t2, t3, t4, t5, t6])
    test_loader_all = torch.utils.data.DataLoader(
        comb_test, batch_size=args.test_batch_size, shuffle=True, **kwargs)
    test_loader1 = torch.utils.data.DataLoader(
        t1, batch_size=args.test_batch_size, shuffle=True, **kwargs)
    test_loader2 = torch.utils.data.DataLoader(
        t2, batch_size=args.test_batch_size, shuffle=True, **kwargs)
    test_loader3 = torch.utils.data.DataLoader(
        t3, batch_size=args.test_batch_size, shuffle=True, **kwargs)
    test_loader4 = torch.utils.data.DataLoader(
        t4, batch_size=args.test_batch_size, shuffle=True, **kwargs)
    test_loader5 = torch.utils.data.DataLoader(
        t5, batch_size=args.test_batch_size, shuffle=True, **kwargs)
    test_loader6 = torch.utils.data.DataLoader(
        t6, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader_all, optimizer, epoch)
        print("Testing all scales")
        test(args, model, device, test_loader_all)
        print("Testing 1/4 scale")
        test(args, model, device, test_loader6)
        print("Testing 1/1 scale")
        test(args, model, device, test_loader1)
        print("Testing 2/3 scale")
        test(args, model, device, test_loader3)
        print("Testing 1/2 scale")
        test(args, model, device, test_loader4)
        print("Testing 3/4 scale")
        test(args, model, device, test_loader2)
        print("Testing  1/3 scale")
        test(args, model, device, test_loader5)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")


if __name__ == '__main__':
    main()
