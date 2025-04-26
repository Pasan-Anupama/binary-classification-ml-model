#Splitting the data as 80% for training and 20% for testing

import os       #Interact with OS (To make directories such things)
import shutil   #File copy, cut, delete operations
import random   #Randomly selection

#Paths
dataset_path = "data/petImages"
train_dir = "data/train"
validation_dir = "data/validation"

#create folders
os.makedirs(f"{train_dir}/dogs", exist_ok=True)
os.makedirs(f"{train_dir}/cats", exist_ok=True)
os.makedirs(f"{validation_dir}/dogs", exist_ok=True)
os.makedirs(f"{validation_dir}/cats", exist_ok=True)

#split ratio (80% train and 20% validate/test)
split_ratio = 0.8

for class_name in ["Dog", "Cat"] :
    src_dir = f"{dataset_path}/{class_name}"
    images = os.listdir(src_dir)
    random.shuffle(images)
    split_idx = int(len(images) * split_ratio)
    
    for i, image in enumerate(images):
        src = f"{src_dir}/{image}"
        if i < split_idx:
            dst = f"{train_dir}/{class_name.lower()}s/{image}"
        else:
            dst = f"{validation_dir}/{class_name.lower()}s/{image}"
        shutil.copy(src, dst)
        
print("Dataset organized successfully !")

