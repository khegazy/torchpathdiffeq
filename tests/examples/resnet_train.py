import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import os
from tqdm import tqdm

# Assuming the compatible model definition is saved in a file named 'models.py'
# You would need to have the file from the previous step saved as models.py
# in the same directory as this training script.
from model import resnet_164

def main():
    # --- 1. Argument Parsing ---
    # Sets up command-line arguments to control the training process.
    parser = argparse.ArgumentParser(description='CIFAR-10 Training for ResNet-164')
    parser.add_argument('--seed', type=int, required=True,
                        help='Random seed for model initialization and training.')
    parser.add_argument('--exp_id', type=str, required=True,
                        help='Experiment ID for naming the saved model file.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=164,
                        help='Number of epochs to train (default: 164)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate (default: 0.1)')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory to download CIFAR-10 data')
    parser.add_argument('--save_dir', type=str, default='./saved_models',
                        help='Directory to save trained models')
    args = parser.parse_args()

    print("Training script started with the following arguments:")
    print(args)

    # --- 2. Reproducibility and Device Setup ---
    # Ensure that the results are reproducible and set the device to GPU if available.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 3. Data Preparation ---
    # Define data augmentations and create data loaders for CIFAR-10.
    print("Preparing CIFAR-10 dataset...")

    # Normalization values for CIFAR-10
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

    # Data augmentation for the training set
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    # Only normalization for the validation set
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    val_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=transform_val)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print("Dataset prepared successfully.")

    # --- 4. Model, Loss, and Optimizer Setup ---
    # Initialize the model, define the loss function, optimizer, and learning rate scheduler.
    print(f"Initializing resnet_164 model with seed: {args.seed}")
    # CIFAR-10 has 10 classes
    model = resnet_164(output_classes=10, seed=args.seed).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=1e-4)

    # A common learning rate schedule for training ResNets on CIFAR-10
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[82, 122], gamma=0.1)

    # --- 5. Training Loop ---
    best_val_accuracy = 0.0

    print("\nStarting training...")
    for epoch in range(args.epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Using tqdm for a progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

            # Update progress bar description
            train_pbar.set_postfix({
                'Loss': f'{running_loss/total_train:.4f}',
                'Acc': f'{100.*correct_train/total_train:.2f}%'
            })

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100. * correct_train / total_train

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

                val_pbar.set_postfix({
                    'Loss': f'{val_loss/total_val:.4f}',
                    'Acc': f'{100.*correct_val/total_val:.2f}%'
                })

        validation_loss = val_loss / len(val_loader)
        validation_accuracy = 100. * correct_val / total_val

        # Print epoch summary
        print(f"Epoch {epoch+1}/{args.epochs} Summary | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {validation_loss:.4f}, Val Acc: {validation_accuracy:.2f}%")

        # --- Save Best Model ---
        if validation_accuracy > best_val_accuracy:
            print(f"New best validation accuracy: {validation_accuracy:.2f}%. Saving model...")
            best_val_accuracy = validation_accuracy

            # Create save directory if it doesn't exist
            if not os.path.isdir(args.save_dir):
                os.makedirs(args.save_dir)

            save_path = os.path.join(args.save_dir, f'resnet164_cifar10_{args.exp_id}.pt')
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

        # Step the scheduler
        scheduler.step()

    print("\nTraining finished.")
    print(f"Best validation accuracy achieved: {best_val_accuracy:.2f}%")

if __name__ == '__main__':
    main()
