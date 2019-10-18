import torch
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_data_loader(data_dir, image_size, batch_size, workers):
    dataset = datasets.ImageFolder(root=data_dir, transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)


def plot_image(dataloader):
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(utils.make_grid(real_batch[0][:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
