import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 设置随机数种子保证论文可复现
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)

# 以类的方式定义参数，还有很多方法，config文件等等
class Args:
    def __init__(self) -> None:
        self.batch_size = 1
        self.lr = 0.001
        self.epochs = 10
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_train = np.array([-2, -1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 20])
        # self.data_val = np.array([15, 16, 17, 0.1, -3, -4])
        self.data_val = np.random.random_integers(-30, 30, size=100)


args = Args()

# 定义一个简单的全连接
class Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# 定义数据集，判断一个数字是否大于8
class Dataset_num(Dataset):
    def __init__(self, flag='train') -> None:
        self.flag = flag
        assert self.flag in ['train', 'val'], 'not implement!'

        if self.flag == 'train':
            self.data = args.data_train
        else:
            self.data = args.data_val

    def __getitem__(self, index: int):
        val = self.data[index]

        if val > 8:
            label = 1
        else:
            label = 0

        return torch.tensor(label, dtype=torch.long), torch.tensor([val], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.data)

class Toy:
    def __init__(self):
        self.model = Net(1, 32, 16, 2).to(args.device) # 网络参数设置，输入为1，输出为2，即判断一个数是否大于8

        self.train_dataset = Dataset_num(flag='train')
        self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=args.batch_size, shuffle=True)
        self.val_dataset = Dataset_num(flag='val')
        self.val_dataloader = DataLoader(dataset=self.val_dataset, batch_size=args.batch_size, shuffle=True)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)  # , eps=1e-8)

    def train(self, prune=False, prune_thres=0.05):
        for epoch in range(args.epochs):
            self.model.train()
            train_epoch_loss = []
            train_acc = []
            train_epochs_loss = []
            acc, nums = 0, 0
            
            # =========================train=======================
            for idx, (label, inputs) in enumerate(tqdm(self.train_dataloader)):
                inputs = inputs.to(args.device)
                label = label.to(args.device)
                outputs = self.model(inputs)
                self.optimizer.zero_grad()
                loss = self.criterion(outputs, label)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0) #用来梯度裁剪
                self.optimizer.step()
                train_epoch_loss.append(loss.item())
                acc += sum(outputs.max(axis=1)[1] == label).cpu()
                nums += label.size()[0]
            train_epochs_loss.append(np.average(train_epoch_loss))
            train_acc.append(100 * acc / nums)
            print("train acc = {:.3f}%, loss = {}".format(100 * acc / nums, np.average(train_epoch_loss)))
        
            val_acc, val_loss = self.eval()
            print("epoch = {}, valid acc = {:.2f}%, loss = {}".format(epoch, val_acc, val_loss))
            
            if prune:
                self.simple_prune(thres=prune_thres)
                
        self.save_model('./ckpts/model.pth')
        
    def eval(self):
        # =========================val=========================
        with torch.no_grad():
            self.model.eval()
            val_loss_list = []
            acc, nums = 0, 0
            for idx, (label, inputs) in enumerate(tqdm(self.val_dataloader)):
                inputs = inputs.to(args.device)  # .to(torch.float)
                label = label.to(args.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, label)
                val_loss_list.append(loss.item())
                acc += sum(outputs.max(axis=1)[1] == label).cpu()
                nums += label.size()[0]
            
        val_loss = np.average(val_loss_list)
        val_acc = 100 * acc / nums
        return val_acc, val_loss
    
    def check_sparsity(self, module):
        w = module.weight.data
        # print(w)
        thres = 0.05
        for thres in np.linspace(0.00, 0.20, 10):
            mask = torch.where(torch.abs(w)<thres, 1, 0)
            print("Sparsity of layer2 with thres=%f = %f"%(thres, torch.sum(mask)/(w.shape[0]*w.shape[1])))
        
    def pred(self, val):
        model = Net(1, 32, 16, 2)
        model.load_state_dict(torch.load('model.pth'))
        model.eval()
        val = torch.tensor(val).reshape(1, -1).float()
        # 需要转换成相应的输入shape，而且得带上batch_size，因此转换成shape=(1,1)这样的形状
        res = model(val)
        # real: tensor([[-5.2095, -0.9326]], grad_fn=<AddmmBackward0>) 需要找到最大值所在的列数，就是标签
        res = res.max(axis=1)[1].item()
        print("predicted label is {}, {} {} 8".format(res, val.item(), ('>' if res == 1 else '<')))

    def plot(self, train_epochs_loss, valid_epochs_loss):
        # =========================plot==========================
        plt.figure(figsize=(12, 4))
        plt.subplot(121)
        plt.plot(train_epochs_loss[:])
        plt.title("train_loss")
        plt.subplot(122)
        plt.plot(train_epochs_loss, '-o', label="train_loss")
        plt.plot(valid_epochs_loss, '-o', label="valid_loss")
        plt.title("epochs_loss")
        plt.legend()
        plt.show()

    def save_model(self, path):
        # =========================save model=====================
        torch.save(self.model.state_dict(), path)
    
    def simple_prune(self, thres):
        module = self.model.layer2[0]
        print('INFO: Pruning...')
        # print('Weight before pruning:')
        # print(module.weight.data)
        mask = torch.abs(module.weight.data) >= thres
        module.weight.data *= mask.float()
        # print('Weight after pruning:')
        # print(module.weight.data)
       
if __name__ == '__main__':
    toy = Toy()
    
    # prune after training
    print('========== Prune after training ===========')
    toy.train()
    toy.check_sparsity(toy.model.layer2[0])  # the second linear layer
    acc_1, loss_1 = toy.eval()
    toy.simple_prune(thres=0.1)
    acc_2, loss_2 = toy.eval()
    print(acc_1, loss_1, acc_2, loss_2)
    # toy.pred(24)
    # toy.pred(3.14)
    # toy.pred(7.8)  # 这个会预测错误，所以数据量对于深度学习很重要
    
    print('========== Prune with training ===========')
    # prune with training
    toy = Toy()
    toy.train(prune=True, prune_thres=0.1)
    acc_1, loss_1 = toy.eval()
    acc_2, loss_2 = toy.eval()
    print(acc_1, loss_1, acc_2, loss_2)
    
    