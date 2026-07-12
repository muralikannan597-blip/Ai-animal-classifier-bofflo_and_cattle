import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import time
import copy

# Import the data loader function you built in Module 1
from dp import get_dataloaders

# ==========================================
# --- NEW CODE: Step 4 (Training Engine) ---
# ==========================================
def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs=5):
    since = time.time()
    
    # Keep track of the best AI weights and accuracy
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data in batches
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad() # Reset the optimizer

                # Forward pass - only track history if we are in 'train' phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}')

            # Deep copy the model if it beat the previous best accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Accuracy: {best_acc:.4f}')

    # Load the best model weights into the model before returning it
    model.load_state_dict(best_model_wts)
    return model

def main():
    print("--- Step 1: Loading Data ---")
    dataloaders, dataset_sizes, class_names = get_dataloaders()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training will run on: {device}")

    print("\n--- Step 2: Loading and Modifying the Model ---")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(class_names))
    model = model.to(device)

    print("\n--- Step 3: Defining Learning Rules ---")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print("\n--- Step 4: Training the Model ---")
    # This will trigger the loop and train for 5 epochs
    model = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs=5)

    # ==========================================
    # --- NEW CODE: Step 5 (Saving Model) ------
    # ==========================================
    print("\n--- Step 5: Saving the Trained Model ---")
    torch.save(model.state_dict(), 'livestock_model.pth')
    print("SUCCESS! Model saved as 'livestock_model.pth' inside your backend folder.")

if __name__ == '__main__':
    main()
