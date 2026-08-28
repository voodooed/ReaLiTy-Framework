import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch

class LidarDataset(Dataset):
    def __init__(self, lidar_real_dir, lidar_sim_adverse_dir,
                 lidar_transform=None, incidence_transform=None, reflectance_transform=None,
                 intensity_sim_transform=None, intensity_real_transform=None):


        # Directories for real and simulated LiDAR data
        self.lidar_real_dir = lidar_real_dir
        self.lidar_sim_adverse_dir = lidar_sim_adverse_dir
        #self.lidar_sim_clear_dir = lidar_sim_clear_dir

        # Listing images in directories
        self.lidar_real_images = os.listdir(lidar_real_dir)
        self.lidar_sim_adverse_images = os.listdir(lidar_sim_adverse_dir)
        #self.lidar_sim_clear_images = os.listdir(lidar_sim_clear_dir)

        # Transformations
        self.lidar_transform = lidar_transform
        self.incidence_transform = incidence_transform
        self.reflectance_transform = reflectance_transform
        self.intensity_sim_transform = intensity_sim_transform
        self.intensity_real_transform = intensity_real_transform

        # Dataset length is the maximum of the number of real and simulated images
        self.length_dataset = max(len(self.lidar_real_images),
                                  len(self.lidar_sim_adverse_images)
                                  )

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):

        # Load paths for real and simulated LiDAR data
        lidar_real_path = os.path.join(self.lidar_real_dir, self.lidar_real_images[index])
        lidar_sim_adverse_path = os.path.join(self.lidar_sim_adverse_dir, self.lidar_sim_adverse_images[index])
        #lidar_sim_clear_path = os.path.join(self.lidar_sim_clear_dir, self.lidar_sim_clear_images[index])


        # Load the .npy files (assume they contain LiDAR data with multiple channels)
        data_real = np.load(lidar_real_path).astype(np.float32)
        data_sim_adverse = np.load(lidar_sim_adverse_path).astype(np.float32)
        #data_sim_clear = np.load(lidar_sim_clear_path).astype(np.float32)

        data_real = np.nan_to_num(data_real, nan=0.0, posinf=0.0, neginf=0.0)
        data_sim_adverse = np.nan_to_num(data_sim_adverse, nan=0.0, posinf=0.0, neginf=0.0)



        #Real Adverse Weather Intensity #CADC
        #lidar_real_range = data_real[0, :, :]         # First channel: LiDAR (range) data
        lidar_real_intensity = data_real[1, :, :]     # Second channel: Real Intensity #CADC for Snow

        #Sim Adverse Weather Range and Intensity
        lidar_sim_adverse_range = data_sim_adverse[0, :, :]
        lidar_sim_adverse_incidence = data_sim_adverse[1, :, :]
        lidar_sim_adverse_reflectance = data_sim_adverse[2, :, :]
        lidar_sim_adverse_intensity  = data_sim_adverse[3, :, :] #Phy Intensity

        #Sim Clear Weather IA and MR
        #lidar_sim_clear_range = data_sim_clear[0, :, :]
        #lidar_sim_clear_incidence = data_sim_clear[1, :, :]
        #lidar_sim_clear_reflectance = data_sim_clear[2, :, :]
        #lidar_sim_clear_intensity = data_sim_clear[3, :, :]  # Fourth channel: Intensity

        # Apply transformations (if provided)
        if self.lidar_transform:
            lidar_sim_adverse_range = self.lidar_transform(lidar_sim_adverse_range)


        if self.incidence_transform:
            lidar_sim_adverse_incidence = self.incidence_transform(lidar_sim_adverse_incidence)

        if self.reflectance_transform:
            lidar_sim_adverse_reflectance = self.reflectance_transform(lidar_sim_adverse_reflectance)

        if self.intensity_sim_transform:
            lidar_sim_adverse_intensity = self.intensity_sim_transform(lidar_sim_adverse_intensity)

        if self.intensity_real_transform:
            lidar_real_intensity = self.intensity_real_transform(lidar_real_intensity)




        concatenated_sim = torch.cat((lidar_sim_adverse_range, lidar_sim_adverse_incidence ,lidar_sim_adverse_reflectance), dim=0)

        return concatenated_sim, lidar_real_intensity, lidar_sim_adverse_intensity
