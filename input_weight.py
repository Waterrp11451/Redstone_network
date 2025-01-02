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
    input_number=str(32768+int(num))
    block="minecraft:command_block{SuccessCount:"+input_number+"}"
    schem.setBlock((x,y,z), block)
def positive_number(num, x, y, z):
    input_number=str(int(num))
    block="minecraft:command_block{SuccessCount:"+input_number+"}"
    schem.setBlock((x, y, z), block)
#放置方块
for up_index, up_item in enumerate(weights_list[0]):
    sum_negative=0
    sum_positive=0
    for index, item in enumerate(up_item):
        index_offset=index//28
        index_offset2=index%28
        index_offset_even=int(index_offset/2)
        index_offset_odd=int((index_offset-1)/2)
        if index_offset%2==0:#如果该数处于第一组
            if item>=0:
                positive_number(item, 0-13*index_offset_even, 0+4*up_index, 0-2*index_offset2)
                sum_positive=sum_positive+item
            if item<0:
                negative_number(item, 0-13*index_offset_even, 0+4*up_index, 0-2*index_offset2)
                sum_negative=sum_negative+item
        if index_offset%2==1:#如果该数处于第二组
            if item>=0:
                positive_number(item, -6-13*index_offset_odd, 0+4*up_index, 1-2*index_offset2)
                sum_positive = sum_positive + item
            if item<0:
                negative_number(item, -6-13*index_offset_odd, 0+4*up_index, 1-2*index_offset2)
                sum_negative = sum_negative + item
        print(f"Index: {index}, Item: {item},Up_index:{up_index}")
    print("Sum of positive:",sum_positive)
    print("Sum of negative:", sum_negative)
schem.save("myschems", "Weights", mcschematic.Version.JE_1_18_2)









