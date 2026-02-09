import os
import argparse


def args_parser():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--data_root', default='data/crowdsat', type=str)
    parser.add_argument('--num_classes', default=2, type=int)

    # training parameters
    parser.add_argument('--epoch', default=100, type=int, metavar='N')
    parser.add_argument('--batch_size', default=16, type=int, metavar='N')
    parser.add_argument('--lr', default=0.0003, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--num_worker', default=6, type=int)

    parser.add_argument('--bilinear', default=True, type=bool)

    # device
    parser.add_argument('--device', default='cuda', type=str)

    # logging
    parser.add_argument('--output_path', default='weights', type=str)
    parser.add_argument('--save', default=None, type=str)
    parser.add_argument('--print_freq', default=8, type=int)

    # val/checkpoint strategy
    parser.add_argument('--checkpoint', default='best', type=str,
                        choices=['best', 'all', 'latest'])
    parser.add_argument('--select_mode', default='max', type=str,
                        choices=['min', 'max'])
    parser.add_argument('--validate_on', default='f1_score', type=str,
                        choices=['f1_score', 'recall', 'precision', 'accuracy', 'mAP'])

    # dataset processing settings(*ptm/ds, dense map -> HxW)
    parser.add_argument('--radius', default=2, type=int)
    parser.add_argument('--ptm_down_ratio', default=1, type=int)
    parser.add_argument('--lmds_kernel_size', default=3, type=int)
    parser.add_argument('--lmds_adapt_ts', default=0.5, type=float)

    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    if args.save is None:
        args.save = os.path.join(args.output_path, 'best_model.pth')
    else:
        args.save = os.path.join(args.output_path, args.save)

    return args