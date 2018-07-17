from __future__ import print_function

import os
import numpy as np
from PIL import Image

import torch.utils.data as data
import torch
import torchvision
import torchvision.transforms as transforms


__all__ = ['origin_data', 'triplet_data']


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.1307], [0.3081])
])

# transform = {
#    'train': transforms.Compose([
#        transforms.RandomCrop(32, padding=4),
#        transforms.RandomHorizontalFlip(),
#        transforms.ToTensor(),
#        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#    ]),
#    'test': transforms.Compose([
#        transforms.ToTensor(),
#        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#    ])
#}

train_sets = torchvision.datasets.MNIST('./data', True, transform, download=False)
tests_sets = torchvision.datasets.MNIST('./data', False, transform, download=False)

# train_sets = torchvision.datasets.CIFAR10('./data/CIFAR10', True, transform['train'], download=False)
# tests_sets = torchvision.datasets.CIFAR10('./data/CIFAR10', False, transform['test'], download=False)

train_loader = torch.utils.data.DataLoader(train_sets, batch_size=32, shuffle=True, num_workers=2)
tests_loader = torch.utils.data.DataLoader(tests_sets, batch_size=32, shuffle=True, num_workers=2)


def origin_data():
    return train_sets, tests_sets, train_loader, tests_loader


class TRIPLET_MNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, ntriplets=50000, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))
        self.triplets = self._make_triplets(ntriplets=ntriplets)

    def _make_triplets(self, ntriplets):
        assert ntriplets > 0
        if self.train:
            labels = self.train_labels.numpy()
        else:
            labels = self.test_labels.numpy()
        triplets = []
        for classes in range(10):
            anc = np.random.choice(np.where(labels == classes)[0], int(ntriplets/10), replace=True)
            pos = np.random.choice(np.where(labels == classes)[0], int(ntriplets/10), replace=True)
            while np.any((anc - pos) == 0):
                np.random.shuffle(pos)
            neg = np.random.choice(np.where(labels != classes)[0], int(ntriplets/10), replace=True)
        for idx in range(anc.shape[0]):
            triplets.append((int(anc[idx]), int(pos[idx]), int(neg[idx])))
        return triplets

    def __getitem__(self, index):
        idx_anc, idx_pos, idx_neg = self.triplets[index]
        if self.train:
            img_anc = self.train_data[idx_anc]
            img_pos = self.train_data[idx_pos]
            img_neg = self.train_data[idx_neg]
        else:
            img_anc = self.test_data[idx_anc]
            img_pos = self.test_data[idx_pos]
            img_neg = self.test_data[idx_neg]
        img_anc = Image.fromarray(img_anc.numpy(), mode='L')
        img_pos = Image.fromarray(img_pos.numpy(), mode='L')
        img_neg = Image.fromarray(img_neg.numpy(), mode='L')
        # img_anc.unsqueeze_(0)
        # img_pos.unsqueeze_(0)
        # img_neg.unsqueeze_(0)
        if self.transform is not None:
            img_anc = self.transform(img_anc)
            img_pos = self.transform(img_pos)
            img_neg = self.transform(img_neg)

        return img_anc, img_pos, img_neg

    def __len__(self):
        if self.train:
            return len(self.triplets)
        else:
            return len(self.triplets)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


triplet_train_sets = TRIPLET_MNIST('./data/', True, ntriplets=50000, transform=transform, download=False)
triplet_train_loader = torch.utils.data.DataLoader(triplet_train_sets,
                                                   batch_size=32,
                                                   shuffle=True,
                                                   num_workers=2)
triplet_tests_sets = TRIPLET_MNIST('./data/', False, ntriplets=10000, transform=transform, download=False)
triplet_tests_loader = torch.utils.data.DataLoader(triplet_tests_sets,
                                                   batch_size=32,
                                                   shuffle=True,
                                                   num_workers=2)

def triplet_data():
    return triplet_tests_sets, triplet_tests_sets, triplet_train_loader, triplet_tests_loader
