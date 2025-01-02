import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 10)
        self.fc2 = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.log_softmax(self.fc2(x), dim=1)
        return x
    def forward_test(self,x):
        x = torch.nn.functional.relu(self.fc1(x))
        return x
def test_model(net, test_data):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            outputs = net.forward(x.view(-1, 28 * 28))
            _, predicted = torch.max(outputs, 1)
            n_correct += (predicted == y).sum().item()
            n_total += y.size(0)
    accuracy = n_correct / n_total
    print(f'Accuracy of the network on the test images: {accuracy * 100}%')

# 加载模型参数
net = Net()
net.load_state_dict(torch.load('mnist_model_params.pth'))
def round_tensor(x):
    return torch.round(x)
def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor(),transforms.Lambda(round_tensor)])
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=15, shuffle=True)
def modify_weights(net):
    for param in net.parameters():
        param.data = torch.round(param.data * 100)
modify_weights(net)

# 获取测试数据
test_data = get_data_loader(is_train=False)

# 测试模型
test_model(net, test_data)
for (n, (x, _)) in enumerate(test_data):
    if n > 3:
        break
    predict = torch.argmax(net.forward(x[0].view(-1, 28 * 28)))
    plt.figure(n)
    plt.imshow(x[0].view(28, 28))
    plt.title("prediction: " + str(int(predict)))
plt.show()