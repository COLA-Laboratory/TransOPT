import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
from transopt.benchmark.HPO import algorithms
from transopt.benchmark.HPO import datasets

# 定义读取模型的路径
model_path = os.path.expanduser('~/transopt_tmp/output/models/ROBERM_RobCifar10_0/model.pkl')
algorithm_name = 'ROBERM'
dataset_name = 'RobCifar10'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hparams = {
    'batch_size': 64,
    'nonlinear_classifier': False,
    'lr': 0.001,
    'weight_decay': 0.00001
}

dataset = vars(datasets)[dataset_name]()
algorithm_class = algorithms.get_algorithm_class(algorithm_name)
algorithm = algorithm_class(dataset.input_shape, dataset.num_classes, len(dataset), hparams)

# 加载模型
checkpoint = torch.load(model_path, map_location=device)  # 加载模型到指定设备
algorithm.load_state_dict(checkpoint['model_dict'])
algorithm.to(device)
algorithm.eval()  # 设置为评估模式

# 定义数据转换（将图像转换为 Tensor）
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为 Tensor
])

# 加载 CIFAR-10 测试数据集
test_dataset = dataset.test
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# 定义将 Tensor 转换为 PIL 图像的工具
to_pil_image = ToPILImage()

# 选择一个测试样本并进行预测
for images, labels in test_loader:
    
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():  # 禁用梯度计算
        outputs, reconstructed_images = algorithm.predict(images)  # 确保 predict 返回解码图像
        _, predicted = torch.max(outputs, 1)

    # 打印预测结果
    print(f"Predicted Label: {predicted.item()}")

    # 将原图和重构图像转换为 PIL 图像
    original_image_pil = to_pil_image(images.squeeze(0).cpu())  # 原图
    reconstructed_image_pil = to_pil_image(reconstructed_images.squeeze(0).cpu())  # 重构图

    # 使用 matplotlib 显示原图和重构图的对比
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 显示原图
    axes[0].imshow(original_image_pil)
    axes[0].set_title('Original Image')
    axes[0].axis('off')  # 隐藏坐标轴

    # 显示重构图
    axes[1].imshow(reconstructed_image_pil)
    axes[1].set_title(f'Reconstructed Image\nPredicted Label: {predicted.item()}')
    axes[1].axis('off')  # 隐藏坐标轴

    plt.tight_layout()
    plt.savefig('rec.png')

    # 仅显示第一个测试样本
    break