import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from models import *
from utils import *
import torch.optim as optim
from torchvision import utils
import time

def main():
    dataroot = "/home/aioz-interns/Desktop/motorbike_generator/not_sure_if_clean_img/"
    
    workers = 4
    batch_size = 32
    image_size = 128
    nc = 3
    nz = 100
    ngf = 64  # gen num feature
    ndf = 64  # dis num feature
    num_epochs = 2001
    lr = 0.0002
    beta1 = 0.5  # Beta1 hyperparam for Adam optimizers
    ngpu = 1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # model
    netG = Generator(nz=nz, nfilt=ngf, num_classes=3).to(device)
    netD = Discriminator(nfilt=ndf, num_classes=3).to(device)
    weights_init(netG)
    weights_init(netD)

    optimizerG = optim.Adam(netG.parameters(), lr=0.00025, betas=(0.1, 0.99))
    optimizerD = optim.Adam(netD.parameters(), lr=0.00100, betas=(0.1, 0.99))

    lr_schedulerG = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizerG, T_0=num_epochs//20, eta_min=0.00001)
    lr_schedulerD = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizerD, T_0=num_epochs//20, eta_min=0.00004)

    train_loader = get_data_loader(dataroot, image_size, batch_size, workers)
    use_soft_noisy_labels = True
    
    for epoch in range(num_epochs):
        epoch_time = time.perf_counter()

        for ii, (real_images, motor_labels) in enumerate(train_loader):
            if real_images.shape[0]!= batch_size: continue
            
            if use_soft_noisy_labels:
                real_labels = torch.squeeze(torch.empty((batch_size, 1), device=device).uniform_(0.70, 0.95))
                fake_labels = torch.squeeze(torch.empty((batch_size, 1), device=device).uniform_(0.05, 0.15))
                for p in np.random.choice(batch_size, size=np.random.randint((batch_size//8)), replace=False):
                    real_labels[p], fake_labels[p] = fake_labels[p], real_labels[p] # swap labels
            else:
                real_labels = torch.full((batch_size, 1), 1.0, device=device)
                fake_labels = torch.full((batch_size, 1), 0.0, device=device)
            
            ############################
            # (1) Update D network
            ###########################
            netD.zero_grad()

            motor_labels = torch.tensor(motor_labels, device=device)
            real_images = real_images.to(device)
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            
            outputR = netD((real_images, motor_labels))
            fake_images = netG((noise, motor_labels))

            outputF = netD((fake_images.detach(), motor_labels))
            errD = (torch.mean((outputR - torch.mean(outputF) - real_labels) ** 2) + 
                    torch.mean((outputF - torch.mean(outputR) + real_labels) ** 2))/2
            errD.backward(retain_graph=True)
            optimizerD.step()

            ############################
            # (2) Update G network
            ###########################
            netG.zero_grad()
            
            outputF = netD((fake_images, motor_labels))
            errG = (torch.mean((outputR - torch.mean(outputF) + real_labels) ** 2) +
                    torch.mean((outputF - torch.mean(outputR) - real_labels) ** 2))/2
            errG.backward()
            optimizerG.step()
            
            lr_schedulerG.step(epoch)
            lr_schedulerD.step(epoch)

        if epoch % 10 == 0:
            print('%.2fs [%d/%d] Loss_D: %.4f Loss_G: %.4f' % (
                time.perf_counter()-epoch_time, epoch+1, num_epochs, errD.item(), errG.item()))
            
            #save model
            gen_model_name = './debug/' + 'gen' + str(epoch) + '.pth'
            dis_model_name = './debug/' + 'dis' + str(epoch) + '.pth'
            torch.save(netG.state_dict(), gen_model_name)
            torch.save(netD.state_dict(), dis_model_name)


if __name__ == "__main__":
    main()