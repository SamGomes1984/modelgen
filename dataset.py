import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class BackgroundRemovalDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        """
        Args:
            dataset_dir (str): Path to the dataset directory containing 'images' and 'masks' folders
            transform: Optional transform to be applied on images and masks
        """
        self.dataset_dir = dataset_dir
        self.image_dir = os.path.join(dataset_dir, 'images')
        self.mask_dir = os.path.join(dataset_dir, 'masks')
        self.transform = transform
        
        # Get all image files and sort them to maintain order
        self.images = [f for f in os.listdir(self.image_dir) if f.startswith('image')]
        self.images.sort(key=lambda x: int(x.replace('image', '').split('.')[0]))
        
        # Create a mapping of image numbers to mask files
        self.mask_files = {}
        for mask_file in os.listdir(self.mask_dir):
            if mask_file.startswith('mask'):
                try:
                    number = int(mask_file.replace('mask', '').split('.')[0])
                    self.mask_files[number] = mask_file
                except ValueError:
                    continue
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image path
        image_name = self.images[idx]
        img_path = os.path.join(self.image_dir, image_name)
        
        # Get corresponding mask number and path
        image_number = int(image_name.replace('image', '').split('.')[0])
        mask_name = self.mask_files.get(image_number)
        
        if mask_name is None:
            raise FileNotFoundError(f"No mask file found for image number {image_number}")
        
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask 