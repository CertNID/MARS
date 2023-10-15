import numpy as np
import torch
from torchvision.models.resnet import resnet50
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from dataset.datasets import get_normalize_layer
import torch.nn as nn
import argparse       
import os                  
import torch                   
from torch.utils.data import DataLoader, TensorDataset                    
from dataset.datasets import *                
from torch.optim import SGD,Optimizer                     
import torch.optim as optim   
from classifier.autoencoder import Autoeconder, Loss_Contrastive
from classifier.detect import calc_centroid_dis_median_mad, Detector
from train_and_evaluate.train_utils import *
from attack.modified_Loss import Loss_AdaptiveClusteringv2
import time
import datetime
import random
from attack.PGD import generate_atk
import sys
from classifier.subnet import AdaptiveClustering, Loss_AdaptiveClustering
from classifier.classifier import AdaptiveClusteringClassifier
from certify.cert import *
from train_and_evaluate.train_evaluate import *

import tool.parseargs

ARCHITECTURES = ['acid',"cade"]
 
def main():

    print("============================CertNID Project============================")

    if torch.cuda.is_available():
        print('Torch cuda is available')
    else:
        raise Exception('Torch cuda is not available')

    # load arguments
    args, device, n_gpu, indir =tool.parseargs.main()
    print('args: %s' % args)    
    print('indir: %s' % indir)
    print('device: %s' % device)
    print('n_gpu: %s' % n_gpu)
    
    if args.gpu:                                        
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu   

    if not os.path.exists(args.outdir):                
        os.mkdir(args.outdir)

    print("args.outdir:",args.outdir)

    if args.mode == 'train':
        print("============================Train============================")
        
        if (args.model == 'acid' and (args.dataset == "ids18" or args.dataset == "ids18_unnormalized")) : 
            num_classes_classifier = get_num_classes(args.dataset)  
            # seed = 42 # Random seed
            seed = args.seed
            print("args.seed:",args.seed)
            set_seed(seed)
            acidtrain(args,args.dataset,args.model,num_classes_classifier,args.lr,args.batch,args.epochs,args.print_freq,args.outdir,device)
            acidevaluate(args,args.dataset,num_classes_classifier,args.batch,args.outdir,device)
    
        elif (args.model == "cade" and (args.dataset == "newhulk" or args.dataset == 'newinfiltration')):
            seed = args.seed
            print("args.seed:",args.seed)
            set_seed(seed)            
            
            cadetrain(args, args.dataset,args.model,args.lr,args.epochs,args.batch,args.print_freq,args.outdir,device,margin=args.margin)
            evaluate(args,args.dataset,args.batch,args.outdir,device,args.mad_threshold)
            
    elif args.mode == 'certify':
        print("============================Certify============================")

        if args.model == 'acid':
            acidcertify(args.dataset,args.certclass,args.feature_noise_distribution,args.outdir,args.Norm,args.certmethod,args.seed,device,args.table_number)
        
        elif args.model == 'cade':
            cadecertify(args.dataset,args.feature_noise_distribution,args.outdir,args.certclass,args.mad_threshold,args.Norm,args.certmethod,args.seed,device,args.table_number)
                                                    
    elif args.mode == 'attack':
        print("============================Attack============================")
        
        encoder_list = torch.load(os.path.join(indir, "acidcheckpoint-encoder_list"))
        encoder_list = nn.ModuleList([e.to(device) for e in encoder_list])
        kernel_net_list = torch.load(os.path.join(indir, "acidcheckpoint-kernel_net_list"))
        kernel_net_list = nn.ModuleList([n.to(device) for n in kernel_net_list]).to(device)
        classifier = AdaptiveClusteringClassifier(encoder_list, kernel_net_list, device)
        number = get_num_classes(args.dataset)
        opt = optim.Adam(classifier.parameters(), lr=args.lr)
        criterion = Loss_AdaptiveClusteringv2(number)
        x,y = get_dataset(args.dataset,"test")[:2]# our dataset will be a list of x and y thats why [:2] 0 is whole x and 1 is whole y.
        nonbindex = np.where(y != 0)[0]
        x = x[nonbindex].numpy()
        xbad = generate_atk(cleanx=x,model=classifier,criterion=criterion,optimizer=opt,max=x.max(),min=x.min(),shape=x.shape,number=number)
        x= torch.Tensor(xbad).to(device)
        ybad = classifier(x).argmax(1)
        y= y[nonbindex].to(device)
        print(ybad)
        print(torch.sum(y == ybad)/y.size(0))
                
if __name__ == "__main__":
    main()

