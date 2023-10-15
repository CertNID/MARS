import argparse
import yaml
import datetime
from tool  import runid
import os
from dataset.datasets import *  


def main(): 
    
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument("--table_number", type = str, default= "table2")
    parser.add_argument('--dataset', type=str, choices=DATASETS)
    parser.add_argument("--cuda", type = int,default=0)
    # parser.add_argument('--model', type=str, choices=ARCHITECTURES)
    parser.add_argument('--model', type=str, choices=['acid',"cade"])
    
    
    parser.add_argument("--seed",type=int)
    parser.add_argument('--split', type = str, choices= ['train','test'])
    parser.add_argument('--certmethod', type= str,choices=['vanilla_random','certnid',"bars","random_first_order","certnid_gaussian_estimate"])
    #certnid vrs frs bars
    parser.add_argument("--Norm", type = str, default= "l2", choices=["l1","l2","linf"])
    parser.add_argument("--feature_noise_distribution",default="gaussian", type = str, choices=["gaussian", "uni", "lap"],help="For acid, certify with isru uniformly, for cade use isru for beign, isru gaussian artan for ssh brute force, isru gaussian for dos hulk, and isru gaussian arctan for infiltration")
    parser.add_argument('--outdir', type=str, help='folder to save model and training log)', default='/home/huan1932/CertNID/resultsCertNID')
    parser.add_argument('--N0', default=100,type= int, help = 'Number of noised samples for identify cA' )
    parser.add_argument('--N', default= 10000,type= int, help = "Number of samples to use for pa" )
    parser.add_argument('--certclass',type= int, default=0, help= "classes to certify")
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch', default=256, type=int, metavar='N',
                        help='batchsize (default: 256 for image, 512 for ids18)')
    parser.add_argument("--certbatch",type = int, default= 1000, help= "batchsize for cert")
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        help='initial learning rate', dest='lr')
    parser.add_argument('--lr_step_size', type=int, default=30,
                        help='How often to decrease learning by gamma.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--noise_sd', default=1, type=float,
                        help="standard deviation of Gaussian noise for data augmentation")
    parser.add_argument('--gpu', default=None, type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--mad-threshold',default=3.5, type = int, help="mad threshold for cade")
    parser.add_argument('--radius_start',type= int, default=0, help= "radius to start")
    parser.add_argument('--radius_end',type= int, default=1, help= "radius to end")
    parser.add_argument('--radius_step',type= float, default=.2, help= "radius to step")
    parser.add_argument('--margin', default= 10, type= int, help = "margin value for cade.")
    parser.add_argument('--mode', default= None, type= str, choices=["certify","train","attack"], help = "margin value for cade.")
    
    
    args = parser.parse_args()      
    
    device = 'cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu'
    n_gpu = torch.cuda.device_count()                                 
    indir = "/home/huan1932/CertNID/resultsCertNID/trained-model"    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # # get arguments dictionary
    # args = parse_arguments()
    # args_dictionary = vars(args)

    # # from command yaml file import configure dictionary
    # DO_NOT_EXPORT = ['illegalkey'] 
    # args_dictionary = correct_args_dictionary(args,args_dictionary, DO_NOT_EXPORT)

    # # copy arguments dictionary
    # args_dictionary_copy = copy_args_dictionary(args_dictionary, DO_NOT_EXPORT)
    # args_dictionary_copy_yaml = yaml.dump(args_dictionary_copy)     
    
    # check_arguments(args)

    # exp_result_dir = set_exp_result_dir(args)
    # os.makedirs(exp_result_dir, exist_ok=True)

    # # save arguments dictionary as yaml file
    # exp_yaml=open(f'{exp_result_dir}/experiment-command.yaml', "w")    
    # exp_yaml.write(args_dictionary_copy_yaml)

    return args, device, n_gpu, indir


