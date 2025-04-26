#Some imam=ges in the dataset may be broken. This code is to remove those corrupted images and clean the code

from PIL import Image
import os

for folder in ['cats', 'dogs'] :
    for file in os.listdir(f'data/train/{folder}'):
           try: 
               img = Image.open(f'data/train/{folder}/{file}')
           except:
               print(f"Removing corrupted files: {file}")
               os.remove(f'data/train/{folder}/{file}') 