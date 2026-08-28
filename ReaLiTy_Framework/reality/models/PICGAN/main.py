import time
import torch
from utils import save_checkpoint, load_checkpoint
import torch.nn as nn
import torch.optim as optim
import config
from discriminator import Discriminator
from generator import Generator
from train import train_fn
from transform_utils import lidar_transform, incidence_transform,reflectance_transform, intensity_sim_transform, intensity_real_transform
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    save_outputs,
)

# S - Simulated Domain
# R - Real Domain


def main():

    config.ensure_output_dirs()

    disc_S = Discriminator(in_channels=config.IN_CHANNELS_S).to(config.DEVICE) #For classifying simulated image
    disc_R = Discriminator(in_channels=config.IN_CHANNELS_R).to(config.DEVICE) #For classifying real image

    gen_R = Generator(img_channels=config.IN_CHANNELS_S,out_channels=config.IN_CHANNELS_R, num_residuals=9).to(config.DEVICE) #Takes in simulated domain data and produce image from real intensity domain
    gen_S = Generator(img_channels=config.IN_CHANNELS_R,out_channels=config.IN_CHANNELS_S, num_residuals=9).to(config.DEVICE) #Takes in real domain data and produce image from simulated intensity domain

    opt_disc = optim.Adam(
        list(disc_S.parameters()) + list(disc_R.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_R.parameters()) + list(gen_S.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    #Loss
    L1 = nn.L1Loss() #For cycle consistency and identity loss
    mse = nn.MSELoss() #For adversarial loss

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_S,
            gen_S,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_R,
            gen_R,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC_S,
            disc_S,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC_R,
            disc_R,
            opt_disc,
            config.LEARNING_RATE,
        )


    train_loader, val_loader = get_loaders(

        train_lidar_real_dir=config.TRAIN_Lidar_Real_DIR,
        train_lidar_sim_adverse_dir=config.TRAIN_Lidar_Sim_Adverse_DIR,

        val_lidar_real_dir=config.VAL_Lidar_Real_DIR,
        val_lidar_sim_adverse_dir=config.VAL_Lidar_Sim_Adverse_DIR,

        batch_size=config.BATCH_SIZE,

        lidar_transform=lidar_transform,
        incidence_transform=incidence_transform,
        reflectance_transform=reflectance_transform,
        intensity_sim_transform=intensity_sim_transform,
        intensity_real_transform=intensity_real_transform,

        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )



    #float16 training
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    start_time = time.time()
    total_epochs = config.NUM_EPOCHS

    for epoch in range(config.START_EPOCH, config.NUM_EPOCHS):
        epoch_start_time = time.time()

        train_fn(
            disc_S,
            disc_R,
            gen_R,
            gen_S,
            train_loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
        )

        epoch_end_time = time.time()
        elapsed_time = epoch_end_time - epoch_start_time
        remaining_epochs = total_epochs - epoch - 1
        estimated_time_remaining = remaining_epochs * elapsed_time
        print(f'Estimated time remaining for training: {estimated_time_remaining / 3600} hours')

        if config.SAVE_MODEL:
            save_checkpoint(gen_S, opt_gen, epoch, filename=config.CHECKPOINT_GEN_S)
            save_checkpoint(gen_R, opt_gen, epoch, filename=config.CHECKPOINT_GEN_R)
            save_checkpoint(disc_S, opt_disc, epoch, filename=config.CHECKPOINT_DISC_S)
            save_checkpoint(disc_R, opt_disc, epoch, filename=config.CHECKPOINT_DISC_R)


        save_outputs(gen_R, val_loader, epoch, folder=config.OUTPUT_FOLDER)

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total time for training: {total_time / 3600} hours')


if __name__ == "__main__":
    main()