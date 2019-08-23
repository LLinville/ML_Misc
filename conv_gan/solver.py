
import torch
import torchvision
import os
from torch import optim
from torch.autograd import Variable
from modules import Discriminator as Discriminator
# from modules import ResidualSumGenerator as Generator
from modules import ResidualGenerator as Generator
import matplotlib.pyplot as plt
import torch.autograd as autograd
from plotter.plotter import Plotter


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Solver(object):
    def __init__(self, config, data_loader):
        self.generator = None
        self.discriminator = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.z_dim = config.z_dim
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.image_size = config.image_size
        self.data_loader = data_loader
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.sample_size = config.sample_size
        self.lr = config.lr
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.sample_path = config.sample_path
        self.model_path = config.model_path
        self.epoch = config.epoch
        self.build_model()

        self.plotter = Plotter()
        
    def build_model(self):
        """Build generator and discriminator."""
        self.generator = Generator(z_dim=self.z_dim)
        print(count_parameters(self.generator))
        self.discriminator = Discriminator()
        print(count_parameters(self.discriminator))
        self.g_optimizer = optim.Adam(self.generator.parameters(),
                                      self.lr, (self.beta1, self.beta2))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(),
                                      self.lr*1, (self.beta1, self.beta2))

        if self.epoch:
            g_path = os.path.join(self.model_path, 'generator-%d.pkl' % self.epoch)
            d_path = os.path.join(self.model_path, 'discriminator-%d.pkl' % self.epoch)
            g_optim_path = os.path.join(self.model_path, 'gen-optim-%d.pkl' % self.epoch)
            d_optim_path = os.path.join(self.model_path, 'dis-optim-%d.pkl' % self.epoch)
            self.generator.load_state_dict(torch.load(g_path))
            self.discriminator.load_state_dict(torch.load(d_path))
            self.g_optimizer.load_state_dict(torch.load(g_optim_path))
            self.d_optimizer.load_state_dict(torch.load(d_optim_path))

        if torch.cuda.is_available():
            self.generator.cuda()
            self.discriminator.cuda()


        
    def to_variable(self, x):
        """Convert tensor to variable."""
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)
    
    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data
    
    def reset_grad(self):
        """Zero the gradient buffers."""
        self.discriminator.zero_grad()
        self.generator.zero_grad()
    
    def denorm(self, x):
        """Convert range (-1, 1) to (0, 1)"""
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def train(self):
        """Train generator and discriminator."""
        fixed_noise = self.to_variable(torch.randn(self.batch_size, self.z_dim))
        total_step = len(self.data_loader)
        for epoch in range(self.epoch, self.epoch + self.num_epochs) if self.epoch else range(self.num_epochs):
            for i, images in enumerate(self.data_loader):
                if len(images) != self.batch_size:
                    continue

                # self.plotter.draw_kernels(self.discriminator)
                for p in self.discriminator.parameters():
                    p.requires_grad = True
                #===================== Train D =====================#
                images = self.to_variable(images)
                images.retain_grad()
                batch_size = images.size(0)
                noise = self.to_variable(torch.randn(batch_size, self.z_dim))
                
                # Train D to recognize real images as real.
                outputs = self.discriminator(images)
                real_loss = torch.mean((outputs - 1) ** 2)      # L2 loss instead of Binary cross entropy loss (this is optional for stable training)
                # real_loss = torch.mean(outputs - 1)
                # Train D to recognize fake images as fake.
                fake_images = self.generator(noise)
                fake_images.retain_grad()
                outputs = self.discriminator(fake_images)
                fake_loss = torch.mean(outputs ** 2)
                # fake_loss = torch.mean(outputs)

                # gradient penalty
                gp_loss = calc_gradient_penalty(self.discriminator, images, fake_images)

                # Backprop + optimize
                d_loss = fake_loss + real_loss + gp_loss
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()
                if i % 10 == 0:
                    self.plotter.draw_activations(fake_images.grad[0], original=fake_images[0])

                g_losses = []
                for p in self.discriminator.parameters():
                    p.requires_grad = False
                #===================== Train G =====================#
                for g_batch in range(5):
                    noise = self.to_variable(torch.randn(batch_size, self.z_dim))

                    # Train G so that D recognizes G(z) as real.
                    fake_images = self.generator(noise)
                    outputs = self.discriminator(fake_images)
                    g_loss = torch.mean((outputs - 1) ** 2)
                    # g_loss = -torch.mean(outputs)
                    # Backprop + optimize
                    self.reset_grad()
                    g_loss.backward()
                    # if g_loss.item() < 0.5 * d_loss.item():
                    #     break
                    self.g_optimizer.step()

                    g_losses.append("%.3f"%g_loss.clone().item())
                # print the log info
                if (i+1) % self.log_step == 0:
                    print('Epoch [%d/%d], Step[%d/%d], d_real_loss: %.4f, ' 
                          'd_fake_loss: %.4f, gp_loss: %s, g_loss: %s'
                          %(epoch+1, self.num_epochs, i+1, total_step, 
                            real_loss.item(), fake_loss.item(), gp_loss.item(), ", ".join(g_losses)))

                # save the sampled images
                # print((i+1)%self.sample_step)
                if (i) % self.sample_step == 0:
                    print("saving samples")
                    fake_images = self.generator(fixed_noise)
                    if not os.path.exists(self.sample_path):
                        os.makedirs(self.sample_path)
                    torchvision.utils.save_image(self.denorm(fake_images.data), 
                        os.path.join(self.sample_path,
                                     'fake_samples-%d-%d.png' %(epoch+1, i+1)))
            
            # save the model parameters for each epoch
            if epoch % 5 == 0:
                if not os.path.exists(self.model_path):
                    os.mkdir(self.model_path)
                g_path = os.path.join(self.model_path, 'generator-%d.pkl' %(epoch+1))
                d_path = os.path.join(self.model_path, 'discriminator-%d.pkl' %(epoch+1))
                g_optim_path = os.path.join(self.model_path, 'gen-optim-%d.pkl' % (epoch + 1))
                d_optim_path = os.path.join(self.model_path, 'dis-optim-%d.pkl' % (epoch + 1))
                torch.save(self.generator.state_dict(), g_path)
                torch.save(self.discriminator.state_dict(), d_path)
                torch.save(self.g_optimizer.state_dict(), g_optim_path)
                torch.save(self.d_optimizer.state_dict(), d_optim_path)
            
    def sample(self):
        
        # Load trained parameters 
        g_path = os.path.join(self.model_path, 'generator-%d.pkl' % self.num_epochs)
        d_path = os.path.join(self.model_path, 'discriminator-%d.pkl' % self.num_epochs)
        self.generator.load_state_dict(torch.load(g_path))
        self.discriminator.load_state_dict(torch.load(d_path))
        self.generator.eval()
        self.discriminator.eval()
        
        # Sample the images
        noise = self.to_variable(torch.randn(self.sample_size, self.z_dim))
        fake_images = self.generator(noise)
        sample_path = os.path.join(self.sample_path, 'fake_samples-final.png')
        torchvision.utils.save_image(self.denorm(fake_images.data), sample_path, nrow=12)
        
        print("Saved sampled images to '%s'" %sample_path)

def calc_gradient_penalty(netD, real_data, fake_data):
    gpu = 0
    use_cuda = True
    #print real_data.size()
    alpha = torch.rand(64, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)#,
                                     # _grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 5
    return gradient_penalty