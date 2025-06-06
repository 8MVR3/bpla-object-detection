import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from mlflow.models.signature import infer_signature
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. Конфигурация
batch_size = 64
epochs = 5
learning_rate = 0.01

# 2. Датасет и загрузчики
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 3. Простая модель
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.fc1 = nn.Linear(16 * 30 * 30, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


model = SimpleCNN()

# 4. Оптимизатор и функция потерь
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


# 5. Функция для тестирования модели
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = correct / total
    return accuracy


# 6. Основной цикл обучения с логированием в mlflow
with mlflow.start_run():
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("learning_rate", learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        accuracy = test(model, test_loader)

        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        mlflow.log_metric("loss", avg_loss, step=epoch)
        mlflow.log_metric("accuracy", accuracy, step=epoch)

    # 7. Логируем модель в mlflow с input_example и signature
    input_example = torch.randn(1, 3, 32, 32)
    input_example_np = input_example.numpy()

    model.eval()
    with torch.no_grad():
        output_example = model(input_example)

    signature = infer_signature(input_example_np, output_example.numpy())

    mlflow.pytorch.log_model(
        model,
        artifact_path="model",
        input_example=input_example_np,
        signature=signature,
    )
