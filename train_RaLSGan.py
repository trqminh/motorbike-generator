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
    dataroot = "/home/aioz-interns/Downloads/data/training_dataset"

    # Number of workers for dataloader
    workers = 4

    # Batch size during training
    batch_size = 64

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 128

    # Number of channels in the training images. For color images this is 3
    nc = 3

    # Size of z latent vector (i.e. size of generator input)
    nz = 100

    # Size of feature maps in generator
    ngf = 64

    # Size of feature maps in discriminator
    ndf = 64

    # Number of training epochs
    num_epochs = 200

    # Learning rate for optimizers
    lr = 0.0002

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # ** DECLARE NET **

    # Create the generator
    netG = Generator(ngpu, nz, ngf, nc).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    # netG.apply(weights_init)
    netG.load_state_dict(torch.load('./debug/gen160.pth'))

    # Print the model
    # print(netG)
    # fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)
    # print(netG(fixed_noise).shape)

    # Create the Discriminator
    netD = Discriminator(ngpu, nc, ndf).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    # netD.apply(weights_init)
    netD.load_state_dict(torch.load('./debug/dis160.pth'))

    # Print the model
    # print(netD)
    # print(netD(torch.rand(batch_size, 3, image_size, image_size, device=device)).shape)

    # ** TRAIN **

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training Loop

    # Lists to keep track of progress
    img_list = []
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
            label = torch.full((b_size,), real_label, device=device)
            outputR = netD(real_cpu).view(-1)
            D_x = outputR.mean().item()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            outputF = netD(fake.detach()).view(-1)
            D_G_z1 = outputF.mean().item()

            errD = (torch.mean((outputR - torch.mean(outputF) - label) ** 2) +
                    torch.mean((outputF - torch.mean(outputR) + label) ** 2)) / 2
            errD.backward(retain_graph=True)
            optimizerD.step()

            ############################
            # (2) Update G network:
            ###########################
            netG.zero_grad()

            outputF = netD(fake).view(-1)

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

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(utils.make_grid(fake, padding=2, normalize=True))

            iters += 1

        if epoch % 10 == 0:
            # save model
            gen_model_name = './debug/' + 'gen' + str(epoch) + '.pth'
            dis_model_name = './debug/' + 'dis' + str(epoch) + '.pth'
            torch.save(netG.state_dict(), gen_model_name)
            torch.save(netD.state_dict(), dis_model_name)

    # save model
    torch.save(netG.state_dict(), 'new_generator_model.pth')
    torch.save(netG.state_dict(), 'new_dis_model.pth')

    # plot loss
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
