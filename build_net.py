import torch
import torch.nn as nn

from Net import *


def build_net(args, mode="train"):
    print("[Build Net]")
    if mode == "train":
        # Build Net
        net = EEGNet_net.EEGNet()

    else:
        net = EEGNet_net.EEGNet()
        path = f'./result/bcic4_2a/{args.stamp}/{args.train_subject[0]}/checkpoint/{args.epochs}.tar'
        checkpoint = torch.load(path)
        net.load_state_dict(checkpoint['net_state_dict'])

    # Set GPU
    if args.gpu != 'cpu':
        assert torch.cuda.is_available(), "Check GPU"
        if args.gpu == "multi":
            device = args.gpu
            net = nn.DataParallel(net)
        else:
            device = torch.device(f'cuda:{args.gpu}')
            torch.cuda.set_device(device)
        net.cuda()
    # Set CPU
    else:
        device = torch.device("cpu")
    # Print
    print(f"device: {device}")
    print("")

    return net
