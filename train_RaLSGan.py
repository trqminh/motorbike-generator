import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from models import *
from utils import *
import torch.optim as optim
from torchvision import utils


def main():

    # ** ARGUMENT **
    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # Root directory for dataset
    dataroot = "/home/aioz-interns/Downloads/data/training_dataset/"
    
    workers = 4
    batch_size = 16
    image_size = 128
    nc = 3
    nz = 100
    ngf = 64  # gen num feature
    ndf = 64  # dis num feature
    num_epochs = 1000
    lr = 0.0002
    beta1 = 0.5  # Beta1 hyperparam for Adam optimizers
    ngpu = 1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # ** DECLARE NET **

    # Create the generator
    netG = Generator(nz=nz, nfilt=ngf).to(device)
    weights_init(netG)
    # netG.load_state_dict(torch.load('./debug/gen195.pth'))

    # Print the model
    # print(netG)
    # labels = torch.empty(batch_size, dtype=torch.long).random_(1)
    # print(labels)
    # labels = labels.to(device)
    # fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)
    # print(netG((fixed_noise, labels)).shape)
    # exit(0)
    
    # Create the Discriminator
    netD = Discriminator(nfilt=ndf).to(device)
    weights_init(netD)
    # netD.load_state_dict(torch.load('./debug/dis195.pth'))

    # Print the model
    # print(netD)
    # print(netD((torch.rand(batch_size, 3, image_size, image_size, device=device), torch.empty(batch_size, dtype=torch.long).random_(1).to(device))).shape)
    # exit(0)

    # ** TRAIN **
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    real_label = 1

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training Loop

    # Lists to keep track of progress
    # img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    dataloader = get_data_loader(dataroot, image_size, batch_size, workers)

    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            # ** Train with all-real batch
            netD.zero_grad()

            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            classes = torch.empty(b_size, dtype=torch.long).random_(1).to(device)
            label = torch.full((b_size,), real_label, device=device)
            
            outputR = netD((real_cpu, classes))
            D_x = outputR.mean().item()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG((noise, classes))
            outputF = netD((fake.detach(), classes))
            D_G_z1 = outputF.mean().item()

            errD = (torch.mean((outputR - torch.mean(outputF) - label) ** 2) +
                    torch.mean((outputF - torch.mean(outputR) + label) ** 2)) / 2
            errD.backward(retain_graph=True)
            optimizerD.step()

            ############################
            # (2) Update G network:
            ###########################
            netG.zero_grad()

            outputF = netD((fake, classes))

            errG = (torch.mean((outputR - torch.mean(outputF) + label) ** 2) +
                    torch.mean((outputF - torch.mean(outputR) - label) ** 2)) / 2

            errG.backward()
            D_G_z2 = outputF.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            iters += 1

        if epoch % 5 == 0:
            # save model
            gen_model_name = './debug/' + 'gen' + str(epoch) + '.pth'
            dis_model_name = './debug/' + 'dis' + str(epoch) + '.pth'
            torch.save(netG.state_dict(), gen_model_name)
            torch.save(netD.state_dict(), dis_model_name)

    # save model
    torch.save(netG.state_dict(), 'new_generator_model.pth')
    torch.save(netD.state_dict(), 'new_dis_model.pth')

    # plot loss
    # plt.figure(figsize=(10, 5))
    # plt.title("Generator and Discriminator Loss During Training")
    # plt.plot(G_losses, label="G")
    # plt.plot(D_losses, label="D")
    # plt.xlabel("iterations")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    main()
