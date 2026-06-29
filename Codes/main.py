from dataset import get_dataloaders
from models import get_model
from utils import get_device

device = get_device()
print(f'Device: {device}')

train_loader, val_loader, test_loader, class_names, class_weights = get_dataloaders()
print(f'\nNumber of classes: {len(class_names)}')
print(f'Classes: {class_names}')

for name in ['custom_cnn', 'efficientnet_v2_s', 'convnext_tiny']:
    model = get_model(name, len(class_names)).to(device)
    print(f'{name}: OK')