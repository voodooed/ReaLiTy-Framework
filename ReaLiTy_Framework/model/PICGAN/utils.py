import os
import torch
import torchvision
from dataset import LidarDataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import config
from torchvision.utils import save_image
import math


def save_checkpoint(model, optimizer, epoch, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr




def save_outputs(gen, val_loader, epoch, folder, num_images=10):
    gen.eval()

    # Fetch a batch from the validation loader
    sim, real, phy = next(iter(val_loader))
    sim = sim.to(config.DEVICE)
    real = real.to(config.DEVICE)
    phy = phy.to(config.DEVICE) 

    mean_real = 0.0158
    std_real  = 0.0462

    mean_sim_phy = 0.1745
    std_sim_phy  = 0.1515

    with torch.no_grad():
        # Generate fake real images from the simulator input
        fake_real = gen(sim)
        fake_real = (fake_real * std_real) + mean_real # Denormalize generated fake images
        phy  = (phy * std_sim_phy) + mean_sim_phy



    # Create a new directory for this epoch
    epoch_dir = os.path.join(folder, f'epoch_{epoch}')
    os.makedirs(epoch_dir, exist_ok=True)

    # Save each image individually
    for i in range(min(num_images, sim.size(0))):

        # Get the image name from the val_loader dataset
        image_name = os.path.splitext(val_loader.dataset.lidar_sim_adverse_images[i])[0]

        # Create the filenames for fake and true images
        fake_real_image_name = f"{image_name}_generated.png"
        sim_image_name = f"{image_name}_sim.png"

        # Save the images using torchvision's save_image
        save_image(fake_real[i], os.path.join(epoch_dir, fake_real_image_name))
        save_image(phy[i], os.path.join(epoch_dir, sim_image_name))

    gen.train()


def get_loaders(
    train_lidar_real_dir,
    train_lidar_sim_adverse_dir,

    val_lidar_real_dir,
    val_lidar_sim_adverse_dir,

    batch_size,
    lidar_transform, 
    incidence_transform,
    reflectance_transform,
    intensity_sim_transform,
    intensity_real_transform,
    num_workers=4,
    pin_memory=True,
):

    train_ds = LidarDataset(
        lidar_real_dir=train_lidar_real_dir,
        lidar_sim_adverse_dir=train_lidar_sim_adverse_dir,

        lidar_transform=lidar_transform,
        incidence_transform=incidence_transform,
        reflectance_transform=reflectance_transform,
        intensity_sim_transform=intensity_sim_transform,
        intensity_real_transform=intensity_real_transform,
        
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    
    val_ds = LidarDataset(
        lidar_real_dir=val_lidar_real_dir,
        lidar_sim_adverse_dir=val_lidar_sim_adverse_dir,


        lidar_transform=lidar_transform,
        incidence_transform=incidence_transform,
        reflectance_transform=reflectance_transform,
        intensity_sim_transform=intensity_sim_transform,
        intensity_real_transform=intensity_real_transform,
        
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader
