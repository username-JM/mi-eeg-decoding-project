import os
import numpy as np
import torch

from metrics import cal_log
from utils import print_update, createFolder, write_json, print_dict

torch.set_printoptions(linewidth=1000)

class Solver:
    def __init__(self, args, net, train_loader, val_loader, criterion, optimizer, scheduler, log_dict):
        self.args = args
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.log_dict = log_dict

    def train(self):
        log_tmp = {key: [] for key in self.log_dict.keys() if "train" in key}
        self.args.batch_size = 115
        self.net.train()
        for i, data in enumerate(self.train_loader):
            # Load batch data
            inputs, labels = data[0].cuda(), data[1].cuda()

            # Feed-forward
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)


            # Backward
            loss.backward()
            self.optimizer.step()

            # Calculate log
            cal_log(log_tmp, outputs=outputs, labels=labels, loss=loss)

            # Print
            sentence = f"({(i + 1) * self.args.batch_size} / {len(self.train_loader.dataset.X)})"
            for key, value in log_tmp.items():
                sentence += f" {key}: {value[i]:0.3f}"
            print_update(sentence, i)
        print("")
        # Record log
        for key in log_tmp.keys():
            self.log_dict[key].append(np.mean(log_tmp[key]))

    def val(self):
        log_tmp = {key: [] for key in self.log_dict.keys() if "val" in key}
        self.args.batch_size = 29
        self.net.eval()
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                # Load batch data
                inputs, labels = data[0].cuda(), data[1].cuda()

                # Feed-forward
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)

                # Calculate log
                cal_log(log_tmp, outputs=outputs, labels=labels, loss=loss)

            # Record log
            for key in log_tmp.keys():
                self.log_dict[key].append(np.mean(log_tmp[key]))

    def experiment(self):
        print("[Start experiment]")
        prev_epoch = torch.load(self.args.train_cont_path)['epoch'] if self.args.train_cont_path else 0
        total_epoch = self.args.epochs + prev_epoch
        for epoch in range(prev_epoch + 1, total_epoch + 1):
            print(f"Epoch {epoch}/{total_epoch}")
            # Train
            self.train()

            # Validation
            self.val()

            # Print
            print("=>", end=' ')
            for key, value in self.log_dict.items():
                print(f"{key}: {value[epoch - 1]:0.3f}", end=' ')
            print("")
            print(f"=> learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}")

            # Update scheduler
            self.scheduler.step() if self.scheduler else None
            # if self.args.A:
            #     self.scheduler2.step()

            # Save checkpoint
            createFolder(os.path.join(self.args.save_path, "checkpoint"))
            torch.save({
                'epoch': epoch,
                'net_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None
            }, os.path.join(self.args.save_path, f"checkpoint/{epoch}.tar"))
            write_json(os.path.join(self.args.save_path, "log_dict.json"), self.log_dict)
            if epoch >= 6:
                os.remove(os.path.join(self.args.save_path, f"checkpoint/{epoch - 5}.tar"))

        # Save args & log_dict
        self.args.seed = torch.initial_seed()
        self.args.cuda_seed = torch.cuda.initial_seed()
        self.args.acc = np.round(self.log_dict['val_acc'][-1], 3)  # NOTE: 학습 시작할 때 저장하거나, 다른 metric 있는 상황도 고려
        delattr(self.args, 'topo') if hasattr(self.args, 'topo') else None
        delattr(self.args, 'phase') if hasattr(self.args, 'phase') else None
        write_json(os.path.join(self.args.save_path, "args.json"), vars(self.args))

        print("====================================Finish====================================")
        print(self.net, '\n')
        print_dict(vars(self.args))
        print(f"Last checkpoint: {os.path.join(self.args.save_path, 'checkpoint', str(epoch) + '.tar')}")
