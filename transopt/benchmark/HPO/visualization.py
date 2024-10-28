import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision import datasets, transforms
from torchvision.transforms import AutoAugmentPolicy, AutoAugment, RandAugment
import torch

def get_cifar10_data(transform):
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=True)
    images, labels = next(iter(dataloader))
    return images.numpy().reshape(1000, -1), labels.numpy()

# Define transforms
transforms_list = {
    'No Augmentation': transforms.ToTensor(),
    'Random Crop': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ]),
    'Random Horizontal Flip': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'Color Jitter': transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ]),
    'Brightness': transforms.Compose([
        transforms.ColorJitter(brightness=0.5),
        transforms.ToTensor(),
    ]),
    'Solarize': transforms.Compose([
        transforms.RandomSolarize(threshold=128),
        transforms.ToTensor(),
    ]),
    'Shear': transforms.Compose([
        transforms.RandomAffine(degrees=0, shear=15),
        transforms.ToTensor(),
    ]),
}

# Prepare data for all transforms
all_data = []
all_labels = []
for name, transform in transforms_list.items():
    print(f"Processing {name}...")
    data, labels = get_cifar10_data(transform)
    all_data.append(data)
    all_labels.append(np.full(labels.shape, list(transforms_list.keys()).index(name)))

# Combine all data
combined_data = np.vstack(all_data)
combined_labels = np.hstack(all_labels)

# Perform t-SNE on combined data
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(combined_data)

# Visualize results
plt.figure(figsize=(16, 16))
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=combined_labels, cmap='tab10')
plt.title('t-SNE Visualization of CIFAR-10 with Different Augmentations')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')

# Add legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=method, 
                   markerfacecolor=plt.cm.tab10(i/len(transforms_list)), markersize=10)
                   for i, method in enumerate(transforms_list.keys())]
plt.legend(handles=legend_elements, title='Augmentation Methods', loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.savefig('cifar10_augmentations_tsne.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualization complete. Check the output image: cifar10_augmentations_tsne.png")