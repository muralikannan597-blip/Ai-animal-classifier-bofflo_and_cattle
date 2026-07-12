import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir='livestock_dataset', batch_size=32):
    """
    Loads and preprocesses the cattle and buffalo images.
    Can be easily imported into other backend files.
    """
    # 1. Define the rules for cleaning and augmenting the images
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 2. Load the datasets from the folders
    image_datasets = {
        x: datasets.ImageFolder(root=f"{data_dir}/{x}", transform=data_transforms[x])
        for x in ['train', 'val']
    }

    # 3. Create the DataLoaders to feed the AI in batches
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=2)
        for x in ['train', 'val']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names

# This part only runs if you test this file directly
if __name__ == '__main__':
    print("Testing dp.py module...")
    loaders, sizes, classes = get_dataloaders()
    print(f"Successfully loaded classes: {classes}")
    print(f"Train size: {sizes['train']} | Val size: {sizes['val']}")