import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from models import *
from utils import *
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from PIL import Image
import shutil


class LatentVectorDataset(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]


def main():
    # ngpu = 1
    # nz = 100
    # ngf = 128
    # nc = 3
    # device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    
    # netG = Generator(ngpu, nz, ngf, nc).to(device)
    # netG.load_state_dict(torch.load('./new_generator_model.pth'))
    
    # latent_vector_set = torch.randn(10000, nz, 1, 1)
    # latent_vector_dataset = LatentVectorDataset(latent_vector_set)
    
    # data_loader = DataLoader(dataset=latent_vector_dataset, batch_size=64, num_workers=4)
    
    # num_img = 0
    # for x in data_loader:
    #     x = x.to(device)
    #     imgs = netG(x)
    #     imgs = imgs.cpu().detach().numpy()
    #     imgs = np.clip(imgs, 0.0, 1.0)
    #     imgs = np.transpose(imgs, (0, 2, 3, 1))
    
    #     for i in range(imgs.shape[0]):
    #         im = np.uint8(imgs[i] * 255)
    #         im = Image.fromarray(im)
    #         im.save('./new_gen_img/' + str(num_img) + '.png', "PNG")
    #         num_img += 1
    
    #         if num_img % 1000 == 0:
    #             print('{} images'.format(num_img))
    
    # shutil.make_archive('new_gen_img', 'zip', './new_gen_img')

    evaluate('./new_gen_img/*.*')


if __name__ == '__main__':
    main()
