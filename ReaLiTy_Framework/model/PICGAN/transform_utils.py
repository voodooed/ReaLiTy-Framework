from torchvision import transforms


lidar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0965], std=[0.1068])
])



incidence_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7156], std=[0.6352])
])

            
reflectance_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2979], std=[0.2743])
])
    

intensity_sim_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1745], std=[0.1515])
])
    

intensity_real_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0158], std=[0.0462]) #CADC
])

