import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import rank_functions

seed = 42
# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
# torch.cuda.manual_seed_all(seed)


class Args:
    def __init__(self) -> None:
        self.batch_size = 256
        self.lr = 0.001
        self.epochs = 5
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.data_train = np.array([-2, -1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 20])
        self.data_train = np.random.randint(-30, 30, size=100)
        # self.data_val = np.array([15, 16, 17, 0.1, -3, -4])
        self.data_val = np.random.randint(-30, 30, size=100)
        self.prune_config = {
            "module": None,
            "scope": "local",
            "block_num": 16,
            "sparsity": 0.5
        }


args = Args()

# A toy network
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

class ToyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    
# A toy dataset for determining whether a given number is greater than 8 or not
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
        # self.model = Net(1, 32, 16, 2).to(args.device) # input size=1, output size=2
        self.model = ToyCNN()
        self.model = self.model.to(args.device)
        self.train_dataset = torchvision.datasets.MNIST('./datasets/', train=True, download=False,
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))]))
        self.val_dataset = torchvision.datasets.MNIST('./datasets/', train=False, download=False,
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))]))
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=True)
        # self.train_dataset = Dataset_num(flag='train')
        # self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=args.batch_size, shuffle=True)
        # self.val_dataset = Dataset_num(flag='val')
        # self.val_dataloader = DataLoader(dataset=self.val_dataset, batch_size=args.batch_size, shuffle=True)

        self.criterion = nn.CrossEntropyLoss().to(args.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)  # , eps=1e-8)

    def train(self, prune=False):
        for epoch in range(args.epochs):
            self.model.train()
            train_epoch_loss = []
            train_acc = []
            train_epochs_loss = []
            acc, nums = 0, 0
            
            # =========================train=======================
            for idx, (inputs, label) in enumerate(tqdm(self.train_dataloader)):
                inputs = inputs.to(args.device)
                label = label.to(args.device)
                outputs = self.model(inputs)
                self.optimizer.zero_grad()
                loss = self.criterion(outputs, label)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0) # for grad clipping
                self.optimizer.step()
                train_epoch_loss.append(loss.item())
                acc += sum(outputs.max(axis=1)[1] == label).cpu()
                nums += label.size()[0]
            train_epochs_loss.append(np.average(train_epoch_loss))
            train_acc.append(100 * acc / nums)
            print("train acc = {:.3f}%, loss = {}".format(100 * acc / nums, np.average(train_epoch_loss)))
            
            if prune:
                # self.simple_prune(thres=prune_thres)
                print('INFO: pruning ...')
                self.structured_prune(silent=True)        
                       
            val_acc, val_loss = self.eval()
            print("epoch = {}, valid acc = {:.2f}%, loss = {}".format(epoch, val_acc, val_loss))
         
        self.save_model('./ckpts/mnist_cnn_model.pth')
        
    def eval(self):
        # =========================val=========================
        with torch.no_grad():
            self.model.eval()
            val_loss_list = []
            acc, nums = 0, 0
            for idx, (inputs, label) in enumerate(tqdm(self.val_dataloader)):
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
            print("Sparsity of current module with thres=%f = %f"%(thres, torch.sum(mask)/(w.shape[0]*w.shape[1])))
        
    def pred(self, val):
        model = Net(1, 32, 16, 2)
        model.load_state_dict(torch.load('model.pth'))
        model.eval()
        val = torch.tensor(val).reshape(1, -1).float()
        res = model(val)
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
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path), map_location=args.device)
        
    def simple_prune(self, module, thres):
        print('INFO: Pruning...')
        # print('Weight before pruning:')
        # print(module.weight.data)
        mask = (torch.abs(module.weight.data) >= thres)
        module.weight.data *= mask.float()
        # print('Weight after pruning:')
        # print(module.weight.data)
        print('INFO: Finished pruning.')

    def structured_prune(self, silent=True):
        module = getattr(self.model, args.prune_config['module'])
        data = module.weight.data
        sparsity = args.prune_config['sparsity']

        mask = rank_functions.block_rank_fn_local(data, args.prune_config, sparsity, silent=silent)
        mask = mask.to(args.device)
        
        module = getattr(self.model, args.prune_config['module'])
        module.weight.data *= mask


           

        

        
if __name__ == '__main__':
    g_seed = random.randint(0, 100)  # change seed for every program execution
    
    toy = Toy()
    

    args.prune_config = {
            "module": toy.model.fc1,
            "scope": "local",
            "block_num": 16,
            "sparsity": 0.5
    }

    # toy = Toy()
    # # # toy.train(prune=False)
    # toy.load_model('./ckpts/mnist_cnn_model.pth')
    # module = toy.model.fc1
    # toy.check_sparsity(module)  # the second linear layer
    
    # print('========== Prune after training ===========')
    # acc_1, loss_1 = toy.eval()
    # # toy.simple_prune(module=module, thres=0.1)
    # toy.structured_prune(module=module, block_num=40, sparsity=0.9)
    # acc_2, loss_2 = toy.eval()
    # print(f"Before pruning: acc={acc_1}, loss={loss_1}. After pruning: acc={acc_2}, loss={loss_2}")
    
    ################ For Sweeping Pruning Params ##############
    # toy.load_model('./ckpts/mnist_cnn_model.pth')
    # acc_1, loss_1 = toy.eval()
    
    # for i in np.linspace(0.5, 1, 6):
    #     toy.load_model('./ckpts/mnist_cnn_model.pth')
    #     print('========== Prune after training ===========')
    #     print("Sparsity=%f"%i)
    #     toy.structured_prune(module=module, block_num=8, sparsity=i)
    #     acc_2, loss_2 = toy.eval()
    #     print(f"Before pruning: acc={acc_1}, loss={loss_1}. After pruning: acc={acc_2}, loss={loss_2}")
    ################################################################
        
    # print('========== Prune with training ===========')
    # toy = Toy()
    # toy.train(prune=True, prune_module=toy.model.fc1, prune_sparsity=0.5)
    # acc_1, loss_1 = toy.eval()
    # acc_2, loss_2 = toy.eval()
    # print(acc_1, loss_1, acc_2, loss_2)
    
    ################ For Sweeping Pruning Params ##############    
    for i in np.linspace(0.8, 1, 3):
    # for i in [0.9, 1]:
        print('========== Prune with training ===========')
        print("Sparsity=%f"%i)
        toy = Toy()
        args.prune_config = {
            "module": 'fc1',
            "scope": "local",
            "block_num": 16,
            "sparsity": i
        }
        toy.train(prune=True)
    ################################################################
        