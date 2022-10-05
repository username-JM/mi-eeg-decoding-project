import os
import importlib
import itertools
import torch
import torch.nn as nn
import torch.optim as optim

from utils import read_json


def make_solver(args, net, train_loader, val_loader):  # TODO: save path 만들기
    print("[Make solver]")
    # Set criterion
    criterion = set_criterion(args)

    # Set optimizer
    optimizer, scheduler = set_optimizer(args, net)

    # Set metrics
    log_dict = set_metrics(args)
    # Set solver
    module = importlib.import_module(f"Solver.{args.net}_solver")
    solver = module.Solver(args, net, train_loader, val_loader, criterion, optimizer, scheduler, log_dict)

    print("")
    return solver


def set_criterion(args):
    if args.criterion == "MSE":
        criterion = nn.MSELoss()
    elif args.criterion == "CEE":
        criterion = nn.CrossEntropyLoss()
    return criterion


def set_optimizer(args, net):
    if args.opt == "Adam":
        optimizer = optim.Adam(list(net.parameters()), lr=args.lr)
    elif args.opt == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=args.lr)

    # Set scheduler
    if args.scheduler:
        if args.scheduler == 'exp':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
        else:
            scheduler = scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

        return optimizer, scheduler

    return optimizer, None


def set_optimizer_tensor(args, tensor):
    if args.opt == "Adam":
        optimizer = optim.Adam([tensor], lr=args.lr)
    elif args.opt == "SGD":
        optimizer = optim.SGD([tensor], lr=args.lr)
    # Load optimizer  NOTE: 코드 맞는지 확인, scheduler 다르면 어떻게 되는지 확인
    if args.train_cont_path:
        print("continue optimizer")
        optimizer.load_state_dict(torch.load(args.train_cont_path)['optimizer_state_dict'])
    # Set scheduler
    if args.scheduler:
        if args.scheduler == 'exp':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
        elif args.scheduler == 'multi_step':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
        elif args.scheduler == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20,
                                                             threshold=0.1, threshold_mode='abs', verbose=True)
        if args.train_cont_path:
            print("continue scheduler")
            scheduler.load_state_dict(torch.load(args.train_cont_path)['scheduler_state_dict'])
        return optimizer, scheduler

    return optimizer, None


def set_metrics(args):
    if args.train_cont_path:
        print("continue log_dict")
        log_dict = read_json(os.path.join(os.path.dirname(os.path.dirname(args.train_cont_path)), "log_dict.json"))
    else:
        log_dict = {f"{phase}_{metric}": [] for phase in ["train", "val"] for metric in args.metrics}
    return log_dict
