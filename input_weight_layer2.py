import torch
from torch import nn
import mcschematic
# 定义与保存权重时相同的模型结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 10)  # 假设输入是28x28的MNIST图像
        self.fc2 = nn.Linear(10, 10)  # 输出层有10个神经元，对应10个类别

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)
model = Net()
model_path = 'mnist_model_params.pth'
model_weights = torch.load(model_path)
model.load_state_dict(model_weights)
weights_list = []
# 遍历模型的参数
for name, param in model.named_parameters():
    # 将每个权重乘以100并四舍五入
    rounded_weights = (param * 100).round()
    # 将权重输入到列表
    weights_list.append((rounded_weights.tolist()))
schem = mcschematic.MCSchematic()
def negative_number(num,x,y,z):
    input_number=str(16777216+int(num))
    block="minecraft:command_block{SuccessCount:"+input_number+"}"
    schem.setBlock((x,y,z), block)
def positive_number(num, x, y, z):
    input_number=str(int(num))
    block="minecraft:command_block{SuccessCount:"+input_number+"}"
    schem.setBlock((x, y, z), block)
def number_command_block(num,x,y,z):
    input_number=str(int(2**num))
    block = "minecraft:repeating_command_block{SuccessCount:" + input_number + "}"
    schem.setBlock((x, y, z), block)
for m in range(0,10):
    for j in range(0,10):
        for i in range(0,14):
            number_command_block(i,0-4*i,0+4*j,0-7*m)
            number_command_block(i,-1-4*i,0+4*j,2-7*m)
for up_index, up_item in enumerate(weights_list[2]):
    for index, item in enumerate(up_item):
        if item>=0:
            for q in range(0,14):
                positive_number((2**q)*item,0-4*q,2+4*up_index,0-7*index)
        if item<0:
            for q in range(0,14):
                negative_number((2**q)*item,0-4*q,2+4*up_index,0-7*index)
print(weights_list[3])
print(weights_list[0])
print(weights_list[1])
print(weights_list[2])
schem.save("myschems", "Weights2", mcschematic.Version.JE_1_18_2)


