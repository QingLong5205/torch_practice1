import torch.onnx
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import gradio as gr
import matplotlib.pyplot as plt

from src.model import Szy


# 可视化部分
def plot_metrics(train_losses, test_losses, accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.savefig('metrics.png')
    plt.close()
    return 'metrics.png'


def classify_image(image):
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                                torchvision.transforms.ToTensor()])
    image = transform(image).to(device)
    image = torch.reshape(image, (1, 3, 32, 32))
    szy.eval()
    with torch.no_grad():
        outputs = szy(image)
        predicted = outputs.argmax(1)
    class_name = train_data.classes[predicted.item()]
    return class_name


# 定义训练的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="../data", train=True,
                                          transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root="../data", train=False,
                                         transform=torchvision.transforms.ToTensor(), download=True)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)

# 利用 Dataloader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=96)
test_dataloader = DataLoader(test_data, batch_size=96)

# 搭建神经网络
szy = Szy()
szy.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(szy.parameters(), lr=learning_rate)

# 训练的轮数
epoch = 10

# 记录损失和准确率
train_losses = []
test_losses = []
accuracies = []

# 开始训练
for i in range(epoch):
    print("--------第 {} 轮训练开始--------".format(i + 1))
    # 训练步骤开始
    szy.train()
    total_train_loss = 0  # 记录当前epoch的总训练损失
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = szy(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    # 记录当前epoch的平均训练损失
    avg_train_loss = total_train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    print("整体训练集上的loss： {}".format(avg_train_loss))

    # 测试步骤开始
    szy.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = szy(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    # 记录当前epoch的平均测试损失
    avg_test_loss = total_test_loss / len(test_dataloader)
    test_losses.append(avg_test_loss)
    # 记录当前epoch的测试准确率
    accuracy_rate = total_accuracy.item() / test_data_size
    accuracies.append(accuracy_rate)

    print("整体测试集上的loss： {}".format(avg_test_loss))
    print("整体测试集上的正确率： {}".format(accuracy_rate))

torch.save(szy, "../szy.pth")

# Gradio 接口
image_input = gr.Image(type="pil", image_mode="RGB", width=32, height=32)
output_text = gr.Textbox()
output_image = gr.Image(type="filepath")

plot_interface = gr.Interface(fn=lambda: plot_metrics(train_losses, test_losses, accuracies),
                              inputs=[],
                              outputs=output_image)

classify_interface = gr.Interface(fn=classify_image,
                                  inputs=image_input,
                                  outputs=output_text,
                                  live=True)

app = gr.TabbedInterface([plot_interface, classify_interface], ["Plot Metrics", "Classify Image"])
app.launch()
