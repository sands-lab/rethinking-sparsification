import train
import argparse, os

parser = argparse.ArgumentParser(description="Reduce trainer")
parser.add_argument('--task', type=str, help='ML Task to run')
parser.add_argument('--seed', type=int)
parser.add_argument('--reducer', type=str, help='Reducer to use')
parser.add_argument('--comp_ratio', type=float, default=0.01, help='k for Top-k')
parser.add_argument('--thresh', type=float, default=0.1, help='threshold for hard-threshold')
parser.add_argument('--no_memory', default=False, action='store_true', help='whether to use ef or not')
parser.add_argument('--rank', type=int, default=-1, help='rank of the node')
parser.add_argument('--world_size', type=int, default=-1, help='number of nodes')
parser.add_argument('--k_low', type=float, default=0.1, help='lower k in ACCORDION')
parser.add_argument('--k_high', type=float, default=0.99, help='higher k in ACCORDION')
args = parser.parse_args()

########################### WANDB #####################################
# Set wandb variables
train.config['proj_name'] = args.task
train.config['entity'] = os.getenv('WANDB_ENTITY')
# Set the below variables to use wandb
train.config['use_wandb'] = int(os.getenv('USE_WANDB', 0))
train.config['wandbkey'] = os.getenv('WANDB_KEY')
#######################################################################
######################## Distributed Training #########################
# Training in an cluster with muliple nodes each having one GPU
train.config['n_workers'] = args.world_size
train.output_dir = os.getenv('OUTPUT_DIR') 
train.config['rank'] = args.rank
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print("Data directory:", os.getenv('DATA'))        
print("My CUDA device:", int(os.getenv('CUDA_VISIBLE_DEVICES')))
train.config['distributed_init_file'] = os.getenv('DIST_INIT') 
##########################################################################

############################### Task #####################################
train.config['seed']= args.seed
train.config['average_reset_epoch_interval']= 30
train.config['distributed_backend']= "nccl"
train.config['fix_conv_weight_norm']= False
train.config['log_verbosity']= 2
train.config['optimizer_batch_size']= 256 #256*8
train.config['num_epochs'] = 300
train.config['optimizer_learning_rate']= 0.1
train.config['optimizer_conv_learning_rate']= train.config['optimizer_learning_rate']
train.config['optimizer_decay_at_epochs']= [150, 250]
train.config['optimizer_decay_with_factor']= 10.0
train.config['optimizer_mom_before_reduce']= False
train.config['optimizer_momentum_type']= "nesterov"
train.config['optimizer_momentum']= 0.9
train.config['optimizer_scale_lr_with_factor']= (train.config['optimizer_batch_size']/128)*train.config['n_workers']
train.config['optimizer_scale_lr_with_warmup_epochs']= 5
train.config['optimizer_wd_before_reduce']= False
train.config['optimizer_weight_decay_bn']= 0.0
train.config['optimizer_weight_decay_conv']= 0.0001
train.config['optimizer_weight_decay_other']= 0.0001
    ############## ResNet18-CIFAR10##########
if args.task == 'resnet18':
    train.config['task']= "Cifar"
    train.config['task_architecture']= "ResNet18"
    #############################################
    ############## ResNet18-CIFAR100 ##########
elif args.task == 'resnet18_cifar100':
    train.config['task']= "Cifar100"
    train.config['task_architecture']= "resnet18"
    #############################################
    ############## SENet-CIFAR10 ##########
elif args.task == 'senet':
    train.config['task']= "Cifar"
    train.config['task_architecture']= "SENet18"
    #############################################
    ############## SENet-CIFAR10 ##########
elif args.task == 'senet_cifar100':
    train.config['task']= "Cifar100"
    train.config['task_architecture']= "seresnet18"
    #############################################
    ############## GoogleNet-CIFAR10 ##########
elif args.task == 'googlenet':
    train.config['task']= "Cifar"
    train.config['task_architecture']= "Googlenet"
    #############################################
    ############## GoogleNet-CIFAR10 ##########
elif args.task == 'googlenet_cifar100':
    train.config['task']= "Cifar100"
    train.config['task_architecture']= "googlenet"
    #############################################

    ################ WikiText2###################
elif args.task == 'wikitext2':
    train.config['task']= "LanguageModeling"
    train.config['optimizer_batch_size']= 128
    train.config['num_epochs'] = 90
    train.config['optimizer_learning_rate']= 1.25
    train.config['optimizer_conv_learning_rate']= train.config['optimizer_learning_rate']
    train.config['optimizer_decay_at_epochs']= [60, 80]
    train.config['optimizer_momentum']= 0.0 # Vanilla SGD for Wikitext2
    train.config['optimizer_scale_lr_with_factor']= (train.config['optimizer_batch_size']/64)*train.config['n_workers']
    train.config['optimizer_weight_decay_conv']= 0.0
    train.config['optimizer_weight_decay_other']= 0.0
    #############################################

###############################################################################

######################### Reducers ####################################
if args.reducer=='exact':
    ############ Exact ###########
    train.config['optimizer_reducer'] = "ExactReducer"
    train.config['run_name'] = 'exact'+'_' + str(train.config['seed'])
    ############################

elif args.reducer=='thresh':
    ############ Threshold ############
    train.config['optimizer_reducer'] = "ThreshReducer"
    train.config['optimizer_reducer_thresh'] = args.thresh
    train.config['optimizer_memory'] = not args.no_memory
    train.config['run_name'] = 'thr' + '_' + str(train.config['optimizer_reducer_thresh']) + '_mem' + str(train.config['optimizer_memory']) +'_' +str(train.config['seed'])
    ############################
elif args.reducer=='topk':
    ############# Topk ############
    train.config['optimizer_reducer'] = "TopKReducer"
    train.config['optimizer_reducer_compression'] = args.comp_ratio
    train.config['optimizer_memory'] = not args.no_memory
    train.config['run_name'] = 'topk' + '_' + str(train.config['optimizer_reducer_compression']) + '_mem' + str(train.config['optimizer_memory'])+'_'+str(train.config['seed'])
    #############################
elif args.reducer=='gtopk':
    ############# Topk ############
    train.config['optimizer_reducer'] = "GlobalTopKReducer"
    train.config['optimizer_reducer_compression'] = args.comp_ratio
    train.config['optimizer_memory'] = not args.no_memory
    train.config['run_name'] = 'gtopk' + '_' + str(train.config['optimizer_reducer_compression']) + '_mem' + str(train.config['optimizer_memory'])+'_'+str(train.config['seed'])
    #############################
elif args.reducer=='accordiontopk':
    ############# Topk ############
    train.config['optimizer_reducer'] = "AccordionTopKReducer"
    train.config['accordion_k_low'] = args.k_low
    train.config['accordion_k_high'] = args.k_high
    train.config['optimizer_memory'] = not args.no_memory
    train.config['run_name'] = 'acck' + '_' + str(train.config['accordion_k_low'])+'_' + str(train.config['accordion_k_high']) + '_mem' + str(train.config['optimizer_memory'])+'_'+str(train.config['seed'])

########################################################################
# train.log_info = your_function_pointer
# train.log_metric = your_metric_function_pointer
########################################################################

train.main()