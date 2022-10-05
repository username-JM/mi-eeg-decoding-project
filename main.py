from config import arg
from data_loader import data_loader
from build_net import build_net
from make_solver import make_solver
from utils import control_random, timeit
import os

#@timeit
def main():
    # model training
    for i in range(1, 10):
        # Parsing

        args = arg(i - 1)

        # os.mkdir(path)
        # Control randomness
        if args.seed:
            control_random(args)

        args.train_subject[0] = i - 1
        train_loader, val_loader = data_loader(args)

        # Build net
        net = build_net(args)

        # Make solver
        solver = make_solver(args, net, train_loader, val_loader)

        solver.experiment()

    # LRP
    for i in range(1, 10):
        args.train_subject[0] = i - 1

        # load data
        train_loader, val_loader = data_loader(args)

        # Build net
        net = build_net(args, "lrp")

        # Make solver
        solver = make_solver(args, net, train_loader, val_loader)

        solver.run_lrp()


if __name__ == '__main__':
    main()
