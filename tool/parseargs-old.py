import argparse
import yaml
import datetime
from tool  import runid
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description='parse command')
    subparsers = parser.add_subparsers(description='parse subcommand',dest='subcommand')
    
    parser_load = subparsers.add_parser('load',help = 'run command from a ./experiments/yaml file')
    parser_load.add_argument('--config',type=str,default='experiments/experiment-command.yaml')
    parser_run = subparsers.add_parser('run',help = 'run command from the command line')

    for parser_object in [parser_load, parser_run]:    

        #   setup
        parser_object.add_argument('--exp_name',type=str,default=None,help='Name of the experiment dataset and nid model',
            choices=['mlp-nslkdd','resnet-nslkdd','lstm-nslkdd','transformer-nslkdd','lstm-infection'])
        parser_object.add_argument('--clamodel',type=str,default=None,
            choices=['mlp','resnet','lstm','transformer'])
        parser_object.add_argument('--dataset',type=str,default=None,
            choices=['nslkdd','infection'])
        parser_object.add_argument('--taskmode',type=str,default=None,
            choices=['stdtrain', 'advattack','robcertify'])
        
        parser_object.add_argument('--classes_num',type=int,default=None,help='Number of classes of the dataset')
        parser_object.add_argument('--features_num',type=int, default=None, help= 
                                   '''
                                   nslkdd, 41 features
                                   
                                   ''')

        parser_object.add_argument('--batchsize',type=int,default=None,help='Batchsize')
        parser_object.add_argument('--maxepochs_num',type=int,default=None,help='Max epochs number')
        parser_object.add_argument('--save_path',type=str,default='/home/huan1932/NID/carnid/result',help='Output path for saving results')
        parser_object.add_argument('--cpus_num',type=int,default=None,help='Number of CPUs to use')
        parser_object.add_argument('--gpus_num',type=int,default=None,help='Number of GPUS to use')
        parser_object.add_argument('--optimizer', type=str,default=None)  
        parser_object.add_argument('--lr',type=float,default=None,help='Intial learning rate')
        parser_object.add_argument('--lr_schedule', type=str, default=None)
        parser_object.add_argument('--loss', type=str,default=None)  

        parser_object.add_argument('--seed', type=int, default=None)  
        parser_object.add_argument('--scaler', type=str,default=None)  

        #   stdtrain
        parser_object.add_argument('--cla_path',type=str, default=None,help='load path of classifier')

        #   adversarial attack
        parser_object.add_argument('--advattack_type', help='attack method', type=str, default=None, choices=['lbfgs','fgsm','pgd','jsma','cw'])
        parser_object.add_argument('--eps', help='budget of perturbation', type=float, default=None)
        parser_object.add_argument('--stepsize', help='budget of each step perturbation', type=float, default=None)
        parser_object.add_argument('--maxstep_num', help='Max steps number', type=int, default=None)

              

    return parser.parse_args()


def correct_args_dictionary(args,args_dictionary,DO_NOT_EXPORT):      
    if args.subcommand == 'run':
        print('args.subcommand=%s, run the command line' % args.subcommand)

    elif args.subcommand == 'load':                                                    
        print('args.subcommand=%s, load from the yaml config' % args.subcommand)
        config_dictionary = yaml.load(open(args.config))   

        # normalize string format in the yaml file
        for key in config_dictionary:                                                                               
            if type(config_dictionary[key]) == str:
                if config_dictionary[key] == 'true':
                    config_dictionary[key] = True                                                                   
                if config_dictionary[key] == 'false':
                    config_dictionary[key] = False                                                                  
                if config_dictionary[key] == 'null':
                    config_dictionary[key] = None

        # remove keys belong to the DO_NOT_EXPORT list from the configure dictionary
        for key in config_dictionary:
            if key not in DO_NOT_EXPORT:                                                                           
                args_dictionary[key] = config_dictionary[key]
            else:                                                                                                 
                print(f"Please ignore the keys '{key}' from the yaml file !")
        print('args_dictionary from load yaml file =%s' % args_dictionary)      

    elif args.subcommand == None:
        raise Exception('args.subcommand=%s, please input the subcommand !' % args.subcommand)   
    else:
        raise Exception('args.subcommand=%s, invalid subcommand, please input again !' % args.subcommand)

    return args_dictionary

def copy_args_dictionary(args_dictionary,DO_NOT_EXPORT):

    args_dictionary_copy = dict(args_dictionary)                                                                  
    for key in DO_NOT_EXPORT:
        if key in args_dictionary:
            del args_dictionary_copy[key]
    return args_dictionary_copy

def check_arguments(args):
    if  args.exp_name == None:
        raise Exception('args.exp_name=None, please input args.exp_name' % args.exp_name)    
    if args.clamodel == None:
        raise Exception('args.clamodel=None, please input args.clamodel' % args.clamodel)
    if  args.dataset == None:
        raise Exception('args.dataset=None, please input args.dataset' % args.dataset)    
    if args.taskmode == None:                                                                        
        raise Exception('args.taskmode=None, please input args.taskmode' % args.taskmode)
    if  args.classes_num == None:
        raise Exception('args.classes_num=None, please input args.classes_num' % args.classes_num)    
    if  args.features_num == None:
        raise Exception('args.features_num=None, please input args.features_num' % args.features_num)           
    if args.batchsize == None:
        raise Exception('args.batchsize=None, please input args.batchsize' % args.batchsize)
    if  args.maxepochs_num == None:
        raise Exception('args.epochs_num=None, please input args.maxepochs_num' % args.maxepochs_num)    
    if args.cpus_num == None:                                                                        
        raise Exception('args.cpus_num=None, please input args.cpus_num' % args.cpus_num)
    if args.gpus_num == None:                                                                        
        raise Exception('args.gpus_num=None, please input args.gpus_num' % args.gpus_num)
    if args.optimizer == None:                                                                        
        raise Exception('args.optimizer=None, please input args.optimizer' % args.optimizer)       
    if args.lr == None:                                                                        
        raise Exception('args.lr=None, please input args.lr' % args.lr)
    if args.lr_schedule == None:                                                                        
        raise Exception('args.lr_schedule=None, please input args.lr_schedule' % args.lr_schedule)            
    if args.seed == None:                                                                        
        raise Exception('args.seed=None, please input args.seed' % args.seed)
    if args.scaler == None:                                                                        
        raise Exception('args.scaler=None, please input args.scaler' % args.scaler)    
    if args.loss == None:                                                                        
        raise Exception('args.loss=None, please input args.loss' % args.loss)     
    
     
def set_exp_result_dir(args):

    save_path = f'{args.save_path}/{args.seed}'    
    cur=datetime.datetime.utcnow()
    date = f'{cur.year:04d}{cur.month:02d}{cur.day:02d}'
    print("date:",date)

    # exp_result_dir=f'{save_path}/{args.taskmode}/{args.exp_name}/{date}'    

    exp_result_dir=f'{save_path}/{args.taskmode}/{args.exp_name}/{args.classes_num}-classes'
    if args.taskmode == 'stdtrain':
        exp_result_dir=f'{exp_result_dir}/{date}'       
    elif args.taskmode == 'advattack':
        exp_result_dir=f'{exp_result_dir}/{args.advattack_type}/eps{args.eps}-stepsize{args.stepsize}-stepnum{args.maxstep_num}/{date}'    
    elif args.taskmode == 'robcertify':
        exp_result_dir=f'{exp_result_dir}/{args.attack_type}/{date}'    

    # add run id for exp_result_dir
    cur_run_id = runid.GetRunID(exp_result_dir)
    exp_result_dir = os.path.join(exp_result_dir, f'{cur_run_id:05d}')    

    return exp_result_dir


def main(): 
    # get arguments dictionary
    args = parse_arguments()
    args_dictionary = vars(args)

    # from command yaml file import configure dictionary
    DO_NOT_EXPORT = ['illegalkey'] 
    args_dictionary = correct_args_dictionary(args,args_dictionary, DO_NOT_EXPORT)

    # copy arguments dictionary
    args_dictionary_copy = copy_args_dictionary(args_dictionary, DO_NOT_EXPORT)
    args_dictionary_copy_yaml = yaml.dump(args_dictionary_copy)     
    
    check_arguments(args)

    exp_result_dir = set_exp_result_dir(args)
    os.makedirs(exp_result_dir, exist_ok=True)

    # save arguments dictionary as yaml file
    exp_yaml=open(f'{exp_result_dir}/experiment-command.yaml', "w")    
    exp_yaml.write(args_dictionary_copy_yaml)

    return args, exp_result_dir


