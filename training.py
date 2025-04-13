import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import nibabel as nib
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchio as tio
import matplotlib.pyplot as plt
import io
from PIL import Image
import torchvision

class HeartMRIDataset(Dataset):
    def __init__(self, data_dir, transform=None, train=True):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.train = train
        
        # Get all image and label files
        self.images = sorted(list((self.data_dir / ('imagesTr' if train else 'imagesTs')).glob('*.nii.gz')))
        if train:
            self.labels = sorted(list((self.data_dir / 'labelsTr').glob('*.nii.gz')))
        
        # Enhanced transforms using TorchIO for 3D augmentation
        if transform and train:
            self.transform = tio.Compose([
                tio.RandomAffine(
                    scales=(0.9, 1.1),
                    degrees=10,
                    translation=5
                ),
                tio.RandomElasticDeformation(
                    num_control_points=7,
                    max_displacement=7.5
                ),
                tio.RandomBiasField(coefficients=0.2),  # MRI-specific augmentation
                tio.RandomNoise(std=0.1),
                tio.RandomGamma(log_gamma=(-0.3, 0.3))
            ])
        else:
            self.transform = None
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image and label
        img_path = self.images[idx]
        img_nib = nib.load(str(img_path))
        img = img_nib.get_fdata()
        
        if self.train:
            label_path = self.labels[idx]
            label = nib.load(str(label_path)).get_fdata()
        else:
            label = np.zeros_like(img)  # Dummy label for test set

        # Normalize image intensity before resizing
        img = (img - img.mean()) / (img.std() + 1e-8)

        # Target size - using powers of 2 for proper upsampling
        target_size = (128, 128, 32)  # All dimensions are powers of 2
        
        # Create empty arrays for resized data
        resized_img = np.zeros(target_size, dtype=img.dtype)
        resized_label = np.zeros(target_size, dtype=label.dtype)
        
        # Calculate scaling factors
        scale = [t/s for t, s in zip(target_size, img.shape)]
        
        # Simple resize using nearest neighbor interpolation
        for i in range(target_size[0]):
            for j in range(target_size[1]):
                for k in range(target_size[2]):
                    orig_i = min(int(i/scale[0]), img.shape[0]-1)
                    orig_j = min(int(j/scale[1]), img.shape[1]-1)
                    orig_k = min(int(k/scale[2]), img.shape[2]-1)
                    resized_img[i,j,k] = img[orig_i,orig_j,orig_k]
                    if self.train:
                        resized_label[i,j,k] = label[orig_i,orig_j,orig_k]

        # Convert to torch tensors
        img_tensor = torch.FloatTensor(resized_img).unsqueeze(0)  # Add channel dimension
        label_tensor = torch.FloatTensor(resized_label).unsqueeze(0)
        
        if self.transform:
            # Convert to TorchIO subject format
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=img_tensor),
                label=tio.LabelMap(tensor=label_tensor)
            )
            transformed = self.transform(subject)
            img_tensor = transformed.image.data
            label_tensor = transformed.label.data
            
        return img_tensor, label_tensor

def get_transforms():
    """Get data augmentation transforms"""
    return transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1)
    ])

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, log_dir):
    writer = SummaryWriter(log_dir)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        
        # Step the scheduler
        scheduler.step(avg_val_loss)
        
        # Log the learning rate
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), Path(log_dir) / 'best_model.pth')
    
    writer.close()
    return model

def evaluate_model(model, test_loader, device):
    model.eval()
    dice_scores = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Calculate Dice score
            pred = (output > 0.5).float()
            intersection = (pred * target).sum(dim=(1,2,3,4))
            union = pred.sum(dim=(1,2,3,4)) + target.sum(dim=(1,2,3,4))
            dice = (2. * intersection + 1e-5) / (union + 1e-5)
            
            dice_scores.extend(dice.cpu().numpy())

def plot_to_tensorboard(writer, tag, image_batch, label_batch, prediction_batch=None, global_step=0):
    """Add 2D slice visualization to tensorboard"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Get middle slices
    slice_idx = image_batch.shape[2] // 2
    
    # Show original image
    axes[0, 0].imshow(image_batch[0, 0, slice_idx].cpu(), cmap='gray')
    axes[0, 0].set_title('Image')
    axes[0, 0].axis('off')
    
    # Show ground truth
    axes[0, 1].imshow(label_batch[0, 0, slice_idx].cpu(), cmap='red')
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].axis('off')
    
    if prediction_batch is not None:
        axes[0, 2].imshow(prediction_batch[0, 0, slice_idx].cpu().detach(), cmap='red')
        axes[0, 2].set_title('Prediction')
        axes[0, 2].axis('off')
    
    # Show different orientations
    axes[1, 0].imshow(image_batch[0, 0, :, image_batch.shape[3]//2].cpu(), cmap='gray')
    axes[1, 0].set_title('Sagittal')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(image_batch[0, 0, :, :, image_batch.shape[4]//2].cpu(), cmap='gray')
    axes[1, 1].set_title('Coronal')
    axes[1, 1].axis('off')
    
    if prediction_batch is not None:
        overlap = image_batch[0, 0, slice_idx].cpu()
        overlap = plt.cm.gray(overlap)
        pred_mask = prediction_batch[0, 0, slice_idx].cpu().detach()
        overlap[pred_mask > 0.5] = [1, 0, 0, 1]
        axes[1, 2].imshow(overlap)
        axes[1, 2].set_title('Overlay')
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image = torchvision.transforms.ToTensor()(image)
    writer.add_image(tag, image, global_step)
    plt.close()
