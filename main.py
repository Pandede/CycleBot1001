import os
from configparser import ConfigParser
from itertools import chain

import torch
import torchvision.transforms as tf
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from tqdm import tqdm

from model import Generator, Discriminator

Image.MAX_IMAGE_PIXELS = None

cfg = ConfigParser()
cfg.read('./config.ini')

# Parameters
epoch = cfg.getint('default', 'epoch')
save_per_epoch = cfg.getint('default', 'save_per_epoch')
sample_size = cfg.getint('default', 'sample_size')
image_size = cfg.getint('default', 'img_size')
crop_size = cfg.getint('default', 'crop_size')
image_channel = cfg.getint('default', 'img_channel')
batch_size = cfg.getint('default', 'batch_size')

lambda_adv_loss = cfg.getfloat('loss', 'lambda_adv_loss')
lambda_cycle_loss = cfg.getfloat('loss', 'lambda_cycle_loss')
lambda_idt_loss = cfg.getfloat('loss', 'lambda_idt_loss')

domain_a_src = cfg.get('path', 'domain_A_src')
domain_b_src = cfg.get('path', 'domain_B_src')
model_src = cfg.get('path', 'model_src')
sample_src = cfg.get('path', 'sample_src')

weight_clipping_limit = cfg.getfloat('wgan', 'weight_clipping_limit')

DEVICE = cfg.get('cuda', 'device')

# Dataset
transform = tf.Compose([tf.Resize(image_size),
                        tf.RandomCrop(crop_size),
                        tf.RandomHorizontalFlip(),
                        tf.RandomVerticalFlip(),
                        tf.ToTensor(),
                        tf.Normalize((.5, .5, .5), (.5, .5, .5))])
domain_a_dataset = ImageFolder(domain_a_src, transform=transform)
domain_b_dataset = ImageFolder(domain_b_src, transform=transform)

loader_args = {"batch_size": batch_size,
               "shuffle": True,
               "num_workers": 4,
               "drop_last": True,
               "pin_memory": True}
domain_a_loader = DataLoader(domain_a_dataset, **loader_args)
domain_b_loader = DataLoader(domain_b_dataset, **loader_args)

# Models
generator_a = Generator().to(DEVICE)
discriminator_a = Discriminator().to(DEVICE)
generator_b = Generator().to(DEVICE)
discriminator_b = Discriminator().to(DEVICE)

# Criterion
rec_criterion = torch.nn.MSELoss()

# Optimizer
g_optimizer = torch.optim.RMSprop(chain(generator_a.parameters(), generator_b.parameters()), lr=1e-4)
d_optimizer = torch.optim.RMSprop(chain(discriminator_a.parameters(), discriminator_b.parameters()), lr=1e-4)

for e in range(epoch):
    with tqdm(total=min(len(domain_a_loader), len(domain_b_loader)), ncols=200) as progress_bar:
        for i, ((domain_a_image, _), (domain_b_image, _)) in enumerate(zip(domain_a_loader, domain_b_loader)):
            domain_a_image = domain_a_image.to(DEVICE)
            domain_b_image = domain_b_image.to(DEVICE)

            # Train Discriminator A
            fake_a_image = generator_a(domain_b_image)

            fake_a_score = discriminator_a(fake_a_image)
            real_a_score = discriminator_a(domain_a_image)

            da_loss = torch.mean(fake_a_score) - torch.mean(real_a_score)

            # Train Discriminator B
            fake_b_image = generator_b(domain_a_image)

            fake_b_score = discriminator_b(fake_b_image)
            real_b_score = discriminator_b(domain_b_image)

            db_loss = torch.mean(fake_b_score) - torch.mean(real_b_score)

            d_optimizer.zero_grad()
            (da_loss + db_loss).backward()
            d_optimizer.step()

            for p in discriminator_a.parameters():
                p.data.clamp_(-weight_clipping_limit, weight_clipping_limit)

            for p in discriminator_b.parameters():
                p.data.clamp_(-weight_clipping_limit, weight_clipping_limit)

            # Train Generator A
            # Adversarial Loss
            fake_a_image = generator_a(domain_b_image)
            fake_a_score = discriminator_a(fake_a_image)

            ga_adv_loss = -torch.mean(fake_a_score)

            # Cycle Loss
            rec_b_image = generator_b(fake_a_image)
            ga_cycle_loss = rec_criterion(rec_b_image, domain_b_image)

            # Identity Loss
            ga_idt_loss = rec_criterion(fake_a_image, domain_b_image)

            # Train Generator B
            # Adversarial Loss
            fake_b_image = generator_b(domain_a_image)
            fake_b_score = discriminator_b(fake_b_image)

            gb_adv_loss = -torch.mean(fake_b_score)

            # Cycle Loss
            rec_a_image = generator_a(fake_b_image)
            gb_cycle_loss = rec_criterion(rec_a_image, domain_a_image)

            # Identity Loss
            gb_idt_loss = rec_criterion(fake_b_image, domain_a_image)

            g_optimizer.zero_grad()
            (
                    (ga_adv_loss + gb_adv_loss) * lambda_adv_loss +
                    (ga_cycle_loss + gb_cycle_loss) * lambda_cycle_loss +
                    (ga_idt_loss + gb_idt_loss) * lambda_idt_loss
            ).backward()
            g_optimizer.step()

            progress_bar.set_description('[Epoch %d][Iteration %d]'
                                         '[GA loss: %.4f/%.4f/%.4f, B loss: %.4f/%.4f/%.4f]'
                                         '[DA loss: %.4f, B loss: %.4f]' %
                                         (e, i,
                                          ga_adv_loss.item(), ga_cycle_loss.item(), ga_idt_loss.item(),
                                          gb_adv_loss.item(), gb_cycle_loss.item(), gb_idt_loss.item(),
                                          da_loss.item(), db_loss.item()))
            progress_bar.update()

        # Sampling
        sample = torch.cat((fake_a_image[:sample_size], domain_b_image[:sample_size],
                            fake_b_image[:sample_size], domain_a_image[:sample_size]), 0)
        save_image(sample, os.path.join(sample_src, '%04d.png' % e), nrow=sample_size)

        if e % save_per_epoch == 0:
            # Save models
            torch.save({'epoch': e,
                        'generator_a': generator_a.state_dict(),
                        'generator_b': generator_b.state_dict(),
                        'discriminator_a': discriminator_a.state_dict(),
                        'discriminator_b': discriminator_b.state_dict()},
                       os.path.join(model_src, 'cyclegan.pkl'))
