import argparse
import torch.backends.cudnn as cudnn
import os
import torch
from trainer import trainer_Synapse
from model.swin_cry import cry

parser=argparse.ArgumentParser()
parser.add_argument('--root_path',type=str,
                    default=r'/root/autodl-tmp/Synapse',help='root path')
parser.add_argument('--dataset',type=str,
                        default='Synapse',help='experiment_name')
parser.add_argument('--list_dir',type=str,
                    default=r'/root/autodl-tmp/ISIC',help='list dir')
parser.add_argument('--img_size',type=int,
                    default=224,help='img size')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--checkpoint',default=None)
parser.add_argument('--num_classes',type=str,default=9,
                    help='num classes')
parser.add_argument('--snapshot',type=str,default=r'/root/cry',
                    help='output dir')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--batch_size', type=int,
                    default=9, help='batch_size per gpu')
parser.add_argument('--max_epochs', type=int,
                    default=1001, help='maximum epoch number to train')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
args = parser.parse_args()

if __name__ == '__main__':
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True


    dataset_name = args.dataset
    dataset_config = {
        'ACDC': {
            'root_path': args.root_path,
            'list_dir': r'/root/cry/datasets/ACDC/lists_ACDC',
            'num_classes': 4,
        },
        'Se':{
            'root_path':'/root/cry/datasets/Se',
            'list_dir':'/root/cry/datasets/Se/lists_Se',
            'num_classes':2,
        },
        'Synapse':{
            'root_path':'/root/autodl-tmp/Synapse/train_npz',
            'list_dir':'/root/autodl-tmp/Synapse/lists_Synapse',
            'num_classes':9,
        },
        'ISIC':{
            'root_path':'/root/autodl-tmp/ISIC',
            'list_dir':'/root/autodl-tmp/ISIC',
            'num_classes':2,
        }
    }
        
    if args.batch_size != 24:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']

    if not os.path.exists(args.snapshot):
        os.makedirs(args.snapshot)

    # GPU训练的时候记得加个CUDA
    model = cry(img_size=args.img_size, num_classes=args.num_classes).cuda()

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))

    trainer = {'Synapse': trainer_Synapse(args,model,args.snapshot)}
    trainer[dataset_name](args, model, args.snapshot)


