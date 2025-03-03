from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

basicTransform = transforms.Compose([
    transforms.ToTensor()
])

class ImageDataset(Dataset):
    def __init__(self, images, metadata=None, transform=None, usingPath=False):
        self.images = images
        self.metadata = metadata
        if transform is None:
            self.transform = basicTransform
        else:
            self.transform = transform
        self.usingPath = usingPath

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.usingPath:
            image = Image.open(image).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.metadata is not None:
            return image, self.metadata[idx]
        else:
            return image

def getDataLoader(images, metadata=None, transform=None, usingPath=False, batch_size=32, shuffle=True, num_workers=2):
    dataset = ImageDataset(images, metadata=metadata, transform=transform, usingPath=usingPath)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader