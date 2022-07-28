import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import math
import numpy as np
from spikingjelly.clock_driven import neuron, encoding, functional
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


parser = argparse.ArgumentParser(description='spikingjelly LIF MNIST Training')

parser.add_argument('--device', default='cuda:0', help='运行的设备，例如“cpu”或“cuda:0”\n Device, e.g., "cpu" or "cuda:0"')

#C:\\Users\\31618\\Desktop\\VU\\2021\\Research\\Neuromorphic Computing\\Code\\SpikingJelly\\
parser.add_argument('--dataset-dir', default='Datasets', help='保存MNIST数据集的位置，例如“./”\n Root directory for saving MNIST dataset, e.g., "./"')
parser.add_argument('--log-dir', default='Logs', help='保存tensorboard日志文件的位置，例如“./”\n Root directory for saving tensorboard logs, e.g., "./"')
parser.add_argument('--model-output-dir', default='Models', help='模型保存路径，例如“./”\n Model directory for saving, e.g., "./"')

parser.add_argument('-b', '--batch-size', default=64, type=int, help='Batch 大小，例如“64”\n Batch size, e.g., "64"')
parser.add_argument('-T', '--timesteps', default=100, type=int, dest='T', help='仿真时长，例如“100”\n Simulating timesteps, e.g., "100"')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='学习率，例如“1e-3”\n Learning rate, e.g., "1e-3": ', dest='lr')
parser.add_argument('--tau', default=2.0, type=float, help='LIF神经元的时间常数tau，例如“100.0”\n Membrane time constant, tau, for LIF neurons, e.g., "100.0"')
parser.add_argument('-N', '--epoch', default=100, type=int, help='训练epoch，例如“100”\n Training epoch, e.g., "100"')


class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.log_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.log_weight)

    def forward(self, input):
        return nn.functional.linear(input, self.log_weight.exp())




def main():
    '''
    :return: None

    * :ref:`API in English <lif_fc_mnist.main-en>`

    .. _lif_fc_mnist.main-cn:

    使用全连接-LIF的网络结构，进行MNIST识别。\n
    这个函数会初始化网络进行训练，并显示训练过程中在测试集的正确率。

    * :ref:`中文API <lif_fc_mnist.main-cn>`

    .. _lif_fc_mnist.main-en:

    The network with FC-LIF structure for classifying MNIST.\n
    This function initials the network, starts trainingand shows accuracy on test dataset.
    '''
    
    args = parser.parse_args()
    print("########## Configurations ##########")
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    print("####################################")

    device = args.device
    dataset_dir = args.dataset_dir
    log_dir = args.log_dir
    model_output_dir = args.model_output_dir
    batch_size = args.batch_size 
    lr = args.lr
    T = args.T
    tau = args.tau
    train_epoch = args.epoch

    writer = SummaryWriter(log_dir)

    # 初始化数据加载器
    train_dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir,
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            #torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir,
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([
            #torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )

    train_data_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_data_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )



    # 定义并初始化网络
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(32 * 32 * 3, 384, bias=False), # layer1
        # PositiveLinear(16 * 16, 64),
        neuron.LIFNode(tau=tau),
        # nn.Linear(16 * 16 * 3, 8 * 8 * 3, bias=False),
        # PositiveLinear(8 * 8, 10),
        # neuron.LIFNode(tau=tau),
        #nn.Linear(8 * 8 * 3, 8 * 8, bias=False),
        # PositiveLinear(8 * 8, 10),
        #neuron.LIFNode(tau=tau),
        nn.Linear(384, 10, bias=False),
        neuron.LIFNode(tau=tau)
    )

    net = net.to(device)
    # 使用Adam优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # 使用泊松编码器
    encoder = encoding.PoissonEncoder()
    train_times = 0
    max_test_accuracy = 0

    test_accs = []
    train_accs = []

    for epoch in range(train_epoch):
        print("Epoch {}:".format(epoch))
        print("Training...")
        train_correct_sum = 0
        train_sum = 0
        net.train()
        for img, label in tqdm(train_data_loader):
            img = img.to(device)
            label = label.to(device)
            label_one_hot = F.one_hot(label, 10).float()

            optimizer.zero_grad()

            # 运行T个时长，out_spikes_counter是shape=[batch_size, 10]的tensor
            # 记录整个仿真时长内，输出层的10个神经元的脉冲发放次数
            for t in range(T):
                if t == 0:
                    out_spikes_counter = net(encoder(img).float())
                else:
                    out_spikes_counter += net(encoder(img).float())

            # out_spikes_counter / T 得到输出层10个神经元在仿真时长内的脉冲发放频率
            out_spikes_counter_frequency = out_spikes_counter / T

            # 损失函数为输出层神经元的脉冲发放频率，与真实类别的MSE
            # 这样的损失函数会使，当类别i输入时，输出层中第i个神经元的脉冲发放频率趋近1，而其他神经元的脉冲发放频率趋近0
            loss = F.mse_loss(out_spikes_counter_frequency, label_one_hot)
            loss.backward()
            optimizer.step()

            '''
            for p in net[1].parameters():
                p.data.clamp_(0)
            for p in net[3].parameters():
                p.data.clamp_(0)

            for params in net[1].parameters():
                for param_vector in params:
                    for value in param_vector:
                        if (value < 0):
                            value = math.exp(value) * 0.01

            for params in net[3].parameters():
                for param_vector in params:
                    for value in param_vector:
                        if (value < 0):
                            value = math.exp(value) * 0.01
            '''
                


            # 优化一次参数后，需要重置网络的状态，因为SNN的神经元是有“记忆”的
            functional.reset_net(net)

            # 正确率的计算方法如下。认为输出层中脉冲发放频率最大的神经元的下标i是分类结果
            train_correct_sum += (out_spikes_counter_frequency.max(1)[1] == label.to(device)).float().sum().item()
            train_sum += label.numel()

            train_batch_accuracy = (out_spikes_counter_frequency.max(1)[1] == label.to(device)).float().mean().item()
            writer.add_scalar('train_batch_accuracy', train_batch_accuracy, train_times)
            train_accs.append(train_batch_accuracy)

            train_times += 1
        train_accuracy = train_correct_sum / train_sum

        print("Testing...")
        net.eval()
        with torch.no_grad():
            # 每遍历一次全部数据集，就在测试集上测试一次
            test_correct_sum = 0
            test_sum = 0
            for img, label in tqdm(test_data_loader):
                img = img.to(device)
                for t in range(T):
                    if t == 0:
                        out_spikes_counter = net(encoder(img).float())
                    else:
                        out_spikes_counter += net(encoder(img).float())

                test_correct_sum += (out_spikes_counter.max(1)[1] == label.to(device)).float().sum().item()
                test_sum += label.numel()
                functional.reset_net(net)
            test_accuracy = test_correct_sum / test_sum
            writer.add_scalar('test_accuracy', test_accuracy, epoch)
            test_accs.append(test_accuracy)
            max_test_accuracy = max(max_test_accuracy, test_accuracy)
        print("Epoch {}: train_acc = {}, test_acc={}, max_test_acc={}, train_times={}".format(epoch, train_accuracy, test_accuracy, max_test_accuracy, train_times))
        print()
    
    # 保存模型
    torch.save(net, model_output_dir + "/lif_snn_cifar10.ckpt")
    # 读取模型
    # net = torch.load(model_output_dir + "/lif_snn_mnist.ckpt")


if __name__ == '__main__':
    main()
