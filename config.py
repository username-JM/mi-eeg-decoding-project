import os
import json
import argparse
import datetime

from utils import make_save_path, createFolder, convert_list, str2list_int, str2list, str2dict, print_info, read_json, \
    band_list


def arg(sbj):
    now = datetime.datetime.now()
    parser = argparse.ArgumentParser()
    # Time
    # parser.add_argument('--start_time', default=time.time(), help="Please do not enter any value.")
    parser.add_argument('--date', default=now.strftime('%Y-%m-%d'), help="Please do not enter any value.")
    parser.add_argument('--time', default=now.strftime('%H:%M:%S'), help="Please do not enter any value.")

    # Mode
    parser.add_argument('--mode', default="train",
                        choices=['test', 'debug'])  # NOTE: test 코드 만들어야 함.
    parser.add_argument('--pretrained_net', help="Please enter the name of the network you want.")
    parser.add_argument('--pretrained_path', help="example: ./result/bcic4_2a/test/1/checkpoint/200.tar")
    parser.add_argument('--train_cont_path', help="example: ./result/bcic4_2a/test/1/checkpoint/200.tar")

    # Net
    parser.add_argument('--net', required=True, help='Please enter your name of net.')
    parser.add_argument('--cfg', type=json.loads)  # NOTE

    # Data
    parser.add_argument('--TmpData', type=str2list_int, help="Please enter the data shape you want.")
    parser.add_argument('--dataset', default='bcic4_2a')
    parser.add_argument('--paradigm', default="session", choices=['ind'], help="Please enter ind or not")
    parser.add_argument('--train_subject', help="Please do not enter any value.")
    parser.add_argument('--val_subject', type=int, nargs='+', default=[1])
    parser.add_argument('--band', type=band_list, default=[[0, 42]], help="Please connect it with a comma.")
    parser.add_argument('--chans', default='all', type=str2list, help="Please connect it with a comma.")
    parser.add_argument('--labels', default='all', type=str2list_int, help="Please connect it with a comma.")
    parser.add_argument('--high_order', action='store_true', help='Default is False')
    parser.add_argument('--edge', type=str2dict, help="Please enter like this, plv_50,40,30_corr_35,25_ng_0")
    parser.add_argument('--segment', action='store_true', help='Default is False')
    if parser.parse_known_args()[0].segment:
        parser.add_argument('--window_size', type=int, default=400)
        parser.add_argument('--step', type=int, default=50)
    # parser.add_argument('--A', action='store_true', help="Trainable A")
    # parser.add_argument('--augment', action='store_true', help='Default is False')
    # parser.add_argument('--voting', action='store_true', help='Default is False')
    # parser.add_argument('--segment_select', type=int)

    # Train
    parser.add_argument('--criterion', default='CEE', help="Please enter loss function you want to use.")
    parser.add_argument('--opt', default='Adam', help="Please enter optimizer you want to use.")
    # parser.add_argument('--regularization', '-reg', dest='reg', help='Please enter the regularization and the penalty coefficient. Ex: L1_0.01')
    parser.add_argument('--metrics', default='loss,acc', type=str2list, help="Please connect it with a comma.")
    parser.add_argument('--learning_rate', '-lr', dest='lr', type=float, default=1e-04)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=144)
    parser.add_argument('--scheduler', '-sch')
    if parser.parse_known_args()[0].scheduler == 'exp':
        parser.add_argument('--gamma', type=float, required=True)
    elif parser.parse_known_args()[0].scheduler == 'multi_step':
        parser.add_argument('--milestones', type=str2list_int, required=True)
        parser.add_argument('--gamma', type=float, required=True)


    # Path
    parser.add_argument('--save_path')
    parser.add_argument('--stamp')

    # Miscellaneous
    parser.add_argument('--gpu', default=1, help="multi / 0 / 1 / cpu")
    parser.add_argument('--seed', type=int)
    parser.add_argument('--print_step', default=5, type=int, help="Number of print results per epoch.")
    # parser.add_argument('--subprocess', action='store_true')
    parser.add_argument('--signature', help="To filter the results.")

    # Parsing
    args = parser.parse_args()

    # Set train subject
    if args.mode == 'debug' or args.paradigm == 'session':
        args.train_subject = args.val_subject
    else:
        args.train_subject = list(set(range(1, 10)) - set(args.val_subject))

    # Set save_path
    if args.mode == 'debug':
        args.save_path = f"./result/debug/0"
        createFolder(args.save_path)
    elif args.mode == 'train':
        assert args.stamp is not None, "You Should enter stamp."
        if args.train_cont_path:
            args.save_path = os.path.dirname(os.path.dirname(args.train_cont_path))
        else:
            args.save_path = f"./result/{args.dataset}/{args.stamp}/{sbj}"
            createFolder(args.save_path)

    # Print
    print_info(vars(args))
    return args

# --segment --band=0,4,4,7,7,13,13,30,30,42
