from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10


def test_dataloader_batch():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=16)
    batch = next(iter(loader))

    assert batch is not None
    images, labels = batch
    assert images.shape[0] == 16
    assert labels.shape[0] == 16
