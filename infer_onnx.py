import onnxruntime
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Преобразования (такие же, как при обучении)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Загружаем тестовый датасет
test_dataset = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Загружаем ONNX модель
session = onnxruntime.InferenceSession(
    "onnx_models/model_7cf1ce2759dc46b096f93296ae0177f1.onnx")

# Получаем имя входного тензора
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Инференс
correct = 0
total = 0

for images, labels in test_loader:
    np_images = images.numpy()
    outputs = session.run([output_name], {input_name: np_images})[0]
    predicted = np.argmax(outputs, axis=1)
    correct += (predicted == labels.numpy()).sum()
    total += labels.size(0)

accuracy = correct / total
print(f"Точность ONNX-модели: {accuracy:.4f}")
