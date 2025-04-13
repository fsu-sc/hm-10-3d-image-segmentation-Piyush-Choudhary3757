import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def analyze_dataset(data_dir):
    # Get paths for training images and labels
    data_dir = Path(data_dir)
    train_images = sorted(list((data_dir / 'imagesTr').glob('*.nii.gz')))
    train_labels = sorted(list((data_dir / 'labelsTr').glob('*.nii.gz')))
    test_images = sorted(list((data_dir / 'imagesTs').glob('*.nii.gz')))
    
    print(f"Number of training images: {len(train_images)}")
    print(f"Number of training labels: {len(train_labels)}")
    print(f"Number of test images: {len(test_images)}")
    
    # Load first image to analyze dimensions and spacing
    img = nib.load(str(train_images[0]))
    label = nib.load(str(train_labels[0]))
    
    print("\nImage Information:")
    print(f"Image dimensions: {img.shape}")
    print(f"Voxel spacing: {img.header.get_zooms()}")
    print(f"Data type: {img.get_data_dtype()}")
    
    # Calculate statistics across the dataset
    volumes = []
    for label_path in train_labels:
        label_data = nib.load(str(label_path)).get_fdata()
        volume = np.sum(label_data > 0)  # Count non-zero voxels
        volumes.append(volume)
    
    print("\nSegmentation Statistics:")
    print(f"Mean segmentation volume: {np.mean(volumes):.2f} voxels")
    print(f"Std segmentation volume: {np.std(volumes):.2f} voxels")
    
    # Visualize sample slices
    data = img.get_fdata()
    label_data = label.get_fdata()
    
    # Create figure with three orientations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Find middle slices
    x, y, z = data.shape
    mid_x, mid_y, mid_z = x//2, y//2, z//2
    
    # Axial view (top-down)
    axes[0, 0].imshow(data[mid_x, :, :].T, cmap='gray')
    axes[0, 0].set_title('Axial (Image)')
    axes[1, 0].imshow(label_data[mid_x, :, :].T, cmap='red')
    axes[1, 0].set_title('Axial (Label)')
    
    # Sagittal view (side)
    axes[0, 1].imshow(data[:, mid_y, :].T, cmap='gray')
    axes[0, 1].set_title('Sagittal (Image)')
    axes[1, 1].imshow(label_data[:, mid_y, :].T, cmap='red')
    axes[1, 1].set_title('Sagittal (Label)')
    
    # Coronal view (front)
    axes[0, 2].imshow(data[:, :, mid_z].T, cmap='gray')
    axes[0, 2].set_title('Coronal (Image)')
    axes[1, 2].imshow(label_data[:, :, mid_z].T, cmap='red')
    axes[1, 2].set_title('Coronal (Label)')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('data_visualization.png')
    plt.close()
    
    # Plot distribution of segmentation volumes
    plt.figure(figsize=(10, 6))
    plt.hist(volumes, bins=20)
    plt.title('Distribution of Segmentation Volumes')
    plt.xlabel('Volume (voxels)')
    plt.ylabel('Frequency')
    plt.savefig('volume_distribution.png')
    plt.close()

if __name__ == "__main__":
    data_dir = "Task02_Heart"
    analyze_dataset(data_dir)
