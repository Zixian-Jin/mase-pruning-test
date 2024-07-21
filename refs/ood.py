# by Bo Miao
import time
import logging
import torch.nn as nn
import torch.nn.parallel
from torch.nn import functional as F
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from PIL import Image
import os
from torchstat import stat

from seg_predict import seg_predict
from models import *
from functions import *

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
logging.basicConfig(level=logging.DEBUG)
logging.info('current time is {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

class main(object):
    def __init__(self, args):
        self.args = args
        self.best_prec1 = 0
        self.start_epoch = 0
        assert args.arch in ["", "resnet18", "resnet50"], "unsupported architecture"
        assert args.arch or args.obj, "must specify at least one model"
        self.data_dir = args.dataset
        self.classes, self.classes_num = load_classes(args.class_file)
        self.obj = args.obj
        self.attn = args.attn
        self.obj_length = 150 if args.obj else 0

        self.base_model = load_base_model(args.arch)
        self.obj_model = None
        self.obj_matrix_length = 1024
        if args.obj:
            self.obj_model = OPAM_Module(self.obj_matrix_length, self.obj_length, args.arch) \
                if args.attn else COPM_Module(self.obj_length, args.arch)
            logging.info(self.obj_model)
            #stat(self.obj_model, (self.obj_matrix_length, self.obj_length, 1)) if args.attn else stat(self.obj_model, (22500,1,1))
            self.obj_model = self.obj_model.cuda()
        self.classifier = Classifier_Mix(self.classes_num, args.arch) \
            if self.base_model and self.obj_model else Classifier_Single(self.classes_num, args.arch)
        logging.info(self.classifier)
        self.classifier = self.classifier.cuda()
        # self.seg_model = seg_predict(root=os.path.join(os.getcwd(), "seg"))
        
        if args.checkpoint: 
            self.base_model, self.obj_model, self.classifier, self.start_epoch, self.best_prec1 = \
                load_checkpoint(args.checkpoint, self.base_model, self.obj_model, self.classifier)
        cudnn.benchmark = True

    def eval_model(self, data_dir):
        self.classifier.eval()
        if self.base_model:
            self.base_model.eval()
        if self.obj_model:
            self.obj_model.eval()

        centre_crop = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        correct_list = []
        total_list = []
        for class_name in os.listdir(data_dir):
            correct, count, num_obj = 0, 0, 0
            for img_name in os.listdir(os.path.join(data_dir, class_name)):
                img_dir = os.path.join(data_dir, class_name, img_name)
                img = Image.open(img_dir)
                try:
                    img = centre_crop(img).unsqueeze(0).cuda()
                except:
                    print('Skip loading the single-channel test image: %s'%img_dir)
                    continue
                logit = calculate_logit(img, [img_dir], self.base_model, self.obj_model, self.classifier,
                                        self.obj, self.attn)
                result = classify_step(logit, self.classes)

                if result == class_name:
                    correct += 1
                count += 1
            acc = 100 * correct / float(count)
            logging.info('Accuracy of {} class is {:2.2f}%, sample number is {}'.format(class_name, acc, count))
            correct_list.append(correct)
            total_list.append(count)
        acc = sum(correct_list) / float(sum(total_list))
        logging.info('Average test accuracy is = {:2.2f}%'.format(100 * acc))

        return acc


    def train_model(self):
        train_data = load_data(os.path.join(self.data_dir, 'train'), self.args, train=True)
        criterion = nn.CrossEntropyLoss().cuda()
        if self.obj_model:
            params = list(self.obj_model.parameters()) + list(self.classifier.parameters())
        else:
            params = list(self.classifier.parameters())
        optimizer = torch.optim.SGD(params, self.args.lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)

        for epoch in range(self.start_epoch, self.args.epochs):
            print('starting epoch {}'.format(int(epoch)+1))
            cur_lr = adjust_learning_rate(optimizer, epoch, self.args.lr)
            abs_ckpt, abs_ckpt_best = def_ckpt_name(self.args.arch, self.obj_length, self.classes_num,
                                                    self.attn, self.data_dir.split('/')[-1], self.args.extra)

            if epoch != 0 and epoch % 5 == 0:
                self.base_model, self.obj_model, self.classifier = \
                    reload_model(self.base_model, self.obj_model, self.classifier, abs_ckpt_best)

            self.classifier.train()
            if self.base_model:
                self.base_model.train()
            if self.obj_model:
                self.obj_model.train()

            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            end = time.time()

            for i, (input, target, path) in enumerate(train_data):
                data_time.update(time.time()-end)
                target = target.cuda()
                img = torch.autograd.Variable(input).cuda()
                logit = calculate_logit(img, path, self.base_model, self.obj_model, self.classifier,
                                        self.obj, self.attn)
                loss = criterion(logit, target)    
                precision = accuracy(logit.data, target, topk=(1,))[0]
                losses.update(loss, input.size(0))
                top1.update(precision, input.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                # if i<10 or i % 50 == 0:
                if i<10 or i%5 == 0:
                    logging.info('Train: [Epoch: {0}] [Batch {1}/{2}]\t'
                            'Time {batch_time.val:.3f}s (avg: {batch_time.avg:.3f}s)\t'
                          'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'TrainPrec@1 {top1.val:.3f}% ({top1.avg:.3f}%)\t'.format(
                           epoch+1, i+1, len(train_data), batch_time=batch_time,
                           data_time=data_time, loss=losses, top1=top1))

            acc = self.eval_model(os.path.join(self.data_dir, 'val'))
            # remember best prec@1 and save checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': self.args.arch,
                'model_state_dict': self.base_model.state_dict() if self.base_model else {},
                'obj_state_dict': self.obj_model.state_dict() if self.obj_model else {},
                'classifier_state_dict': self.classifier.state_dict(),
                'best_prec1': self.best_prec1,
            }, acc,  self.best_prec1, abs_ckpt, abs_ckpt_best, self.base_model, self.obj_model)
            self.best_prec1 = max(acc, self.best_prec1)
            print("The best validation accuracy obtained during training is = {}%, "
                  "current lr = {:.4f}".format(self.best_prec1*100, cur_lr))

    def test_model(self, img_dir):
        self.classifier.eval()
        if self.base_model:
            self.base_model.eval()
        if self.obj_model:
            self.obj_model.eval()

        centre_crop = transforms.Compose([
            transforms.Resize((256, 256)),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        #img_dir = os.path.join(os.getcwd(), img_dir)
        img = Image.open(img_dir).convert('RGB')
        img = centre_crop(img).unsqueeze(0).cuda()
        logit = calculate_logit(img, [img_dir], self.base_model, self.obj_model, self.classifier,
                                self.obj, self.attn)
        class_vector = F.softmax(logit, 1).data.squeeze()
        assert len(class_vector) == len(self.classes), "class number must match"
        probs, idx = class_vector.sort(0, True)
        res = []
        prob = []
        probs = probs.cpu().numpy()
        for i, j in enumerate(idx):
            res.append(self.classes[j])
            prob.append(round(probs[i],4))
        logging.info('Test result is {}, corresponding probability is {}'.format(res, prob))


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print(args)
    entrance = main(args)
    if args.eval_only:
        entrance.eval_model(args.dataset)
    elif args.test:
        entrance.test_model(args.dataset)
    else:
        entrance.train_model()