import torch
from pathlib import Path
from mymodel import UNet3D, DiceLoss
from training import HeartMRIDataset, train_model, evaluate_model
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import torchvision

def plot_to_tensorboard(writer, name, data, target, output, step):
    # Select the middle slice for visualization
    slice_idx = data.size(2) // 2
    data_slice = data[0, 0, slice_idx].unsqueeze(0)  # Add batch dimension
    target_slice = target[0, 0, slice_idx].unsqueeze(0)
    output_slice = (output[0, 0, slice_idx] > 0.5).float().unsqueeze(0)
    
    # Stack the images vertically for visualization
    grid = torch.cat([data_slice, target_slice, output_slice], dim=0)
    grid = grid.unsqueeze(1)  # Add channel dimension
    writer.add_images(name, grid, step, dataformats='NCHW')

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data directory
    data_dir = "/home/pc23n/hm-10-3d-image-segmentation-Piyush-Choudhary3757-main/Task02_Heart"
    
    # Create datasets
    train_dataset = HeartMRIDataset(data_dir, train=True)
    val_dataset = HeartMRIDataset(data_dir, train=True)  # We'll split train data for validation
    
    # Split train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    # Initialize model
    model = UNet3D(in_channels=1, out_channels=1).to(device)
    
    # Loss and optimizer
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    
    # Training parameters
    num_epochs = 100
    log_dir = Path("runs/heart_segmentation")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Train model with enhanced visualization
    print("Starting training...")
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
            
            # Add visualizations every 50 batches
            if batch_idx % 50 == 0:
                plot_to_tensorboard(
                    writer,
                    f'train/batch_{batch_idx}',
                    data,
                    target,
                    output,
                    epoch * len(train_loader) + batch_idx
                )
        
        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                
                # Add validation visualizations
                if batch_idx == 0:
                    plot_to_tensorboard(
                        writer,
                        f'validation/epoch_{epoch}',
                        data,
                        target,
                        output,
                        epoch
                    )
        
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
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(log_dir / 'best_model.pth'))
    
    # Create test dataset and loader
    test_dataset = HeartMRIDataset(data_dir, train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()
