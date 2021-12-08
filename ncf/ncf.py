# Copyright (c) 2018, deepakn94, codyaustun, robieta. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# -----------------------------------------------------------------------
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------
#
# Copyright (c) 2021, sands-lab, KAUST.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------

import warnings
import torch.jit
import os
import math
import time
import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn as nn

import utils
import dataloading
from neumf import NeuMF

import gradient_reducers
import wandb


from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import ReduceOp

def parse_args():
    parser = ArgumentParser(description="Train a Nerual Collaborative"
                                        " Filtering model")
    parser.add_argument('--data', type=str,
                        help='Path to test and training data files')
    parser.add_argument('-e', '--epochs', type=int, default=30,
                        help='Number of epochs for training')
    parser.add_argument('-b', '--batch_size', type=int, default=2**20,
                        help='Number of examples for each iteration')
    parser.add_argument('--valid_batch_size', type=int, default=2**20,
                        help='Number of examples in each validation chunk')
    parser.add_argument('-f', '--factors', type=int, default=64,
                        help='Number of predictive factors')
    parser.add_argument('--layers', nargs='+', type=int,
                        default=[256, 256, 128, 64],
                        help='Sizes of hidden layers for MLP')
    parser.add_argument('-n', '--negative_samples', type=int, default=4,
                        help='Number of negative examples per interaction')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.0045,
                        help='Learning rate for optimizer')
    parser.add_argument('-k', '--topk', type=int, default=10,
                        help='Rank for test examples to be considered a hit')
    parser.add_argument('--seed', '-s', type=int, default=1,
                        help='Manually set random seed for torch')
    parser.add_argument('--threshold', '-t', type=float, default=1.0,
                        help='Stop training early at threshold')
    parser.add_argument('--beta1', '-b1', type=float, default=0.25,
                        help='Beta1 for Adam')
    parser.add_argument('--beta2', '-b2', type=float, default=0.5,
                        help='Beta1 for Adam')
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='Epsilon for Adam')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability, if equal to 0 will not use dropout at all')
    parser.add_argument('--checkpoint_dir', default='/data/checkpoints/', type=str,
                        help='Path to the directory storing the checkpoint file')
    parser.add_argument('--load_checkpoint_path', default=None, type=str,
                        help='Path to the checkpoint file to be loaded before training/evaluation')
    parser.add_argument('--mode', choices=['train', 'test'], default='train', type=str,
                        help='Passing "test" will only run a single evaluation, otherwise full training will be performed')
    parser.add_argument('--grads_accumulated', default=1, type=int,
                        help='Number of gradients to accumulate before performing an optimization step')
    parser.add_argument('--shared_path', default='.', type=str,
                        help='a shared path visible to all processes for rendezvous')
    parser.add_argument('--backend', default='nccl', type=str,
                        help='backend for distributed processing')
    parser.add_argument('--log_path', default='log.json', type=str,
                        help='Path for the JSON training log')
    parser.add_argument('--no_memory', default=False, action='store_true', help='whether to use EF or not')
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world_size', type=int, default=8)
    parser.add_argument('--reducer', type=str, default='exact')
    parser.add_argument('--comp_ratio', type=float, default=0.01, help='k in Top-k')
    parser.add_argument('--thresh', type=float, default=0.001, help='threshold in hard threshold')
    ## ACCORDION params ##
    parser.add_argument('--k_low', type=float, default=0.1, help='lower k in ACCORDION')
    parser.add_argument('--k_high', type=float, default=0.99, help='higher k in ACCORDION')
    ######################
    parser.add_argument('--wandbkey', type=str, default=None) # Add wandb key here
    return parser.parse_args()


def init_distributed(args):
    args.distributed = args.world_size >= 1

    if args.distributed:
        print('distributed')
        print("Rank:", args.rank)
        '''
        Set cuda device so everything is done on the right GPU.
        THIS MUST BE DONE AS SOON AS POSSIBLE.
        '''
        torch.cuda.set_device(0)
        args.local_rank = 0
        
        if args.reducer=='exact':
            run_name = args.reducer+'_'+ str(args.seed)
        elif args.reducer=='thresh':
            run_name = args.reducer+'_'+str(args.thresh)+'_'+ str(args.seed)
        elif args.reducer=='topk':
            run_name = args.reducer+'_'+str(args.comp_ratio)+'_'+ str(args.seed)
        elif args.reducer=='gtopk':
            run_name = args.reducer+'_'+str(args.comp_ratio)+'_'+ str(args.seed)
        elif args.reducer=='accordiontopk':
            run_name = 'acck' + '_' + str(args.k_low)+'_' + str(args.k_high) + '_'+ str(args.seed)
        
        filename = "dist_init"+'_'+run_name
        if not os.path.isdir(args.shared_path):
            raise RuntimeError(f"{shared_path} not a valid (existing) directory")
        shared_file = os.path.join(args.shared_path, filename)

        '''Initialize distributed communication'''
        torch.distributed.init_process_group(backend=args.backend, init_method=f'file://{shared_file}', world_size=args.world_size, rank=args.rank)
    else:
        args.local_rank = 0


def val_epoch(model, x, y, dup_mask, real_indices, K, samples_per_user, num_user,
              epoch=None, distributed=False):
    model.eval()

    with torch.no_grad():
        p = []
        for u,n in zip(x,y):
            p.append(model(u, n, sigmoid=True).detach())

        temp = torch.cat(p).view(-1,samples_per_user)
        del x, y, p

        # set duplicate results for the same item to -1 before topk
        temp[dup_mask] = -1
        out = torch.topk(temp,K)[1]
        # topk in pytorch is stable(if not sort)
        # key(item):value(prediction) pairs are ordered as original key(item) order
        # so we need the first position of real item(stored in real_indices) to check if it is in topk
        ifzero = (out == real_indices.view(-1,1))
        hits = ifzero.sum()
        ndcg = (math.log(2) / (torch.nonzero(ifzero)[:,1].view(-1).to(torch.float)+2).log_()).sum()

    if distributed:
        torch.distributed.all_reduce(hits, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(ndcg, op=torch.distributed.ReduceOp.SUM)

    hr = hits.item() / num_user
    ndcg = ndcg.item() / num_user

    model.train()
    return hr, ndcg


def main():
    args = parse_args()
    init_distributed(args)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # sync workers before timing
    if args.distributed:
        torch.distributed.broadcast(torch.tensor([1], device="cuda"), 0)
    torch.cuda.synchronize()
    
    print("World size:", torch.distributed.get_world_size())
    print ('Available devices:', torch.cuda.device_count())
    print ('Current cuda device:', torch.cuda.current_device())
    
    timer = Timer(verbosity_level=3, log_fn=metric)
    device = torch.device('cuda:{}'.format(args.local_rank))
    
    if args.reducer=='exact':
        reducer = getattr(gradient_reducers, "ExactReducer")(1, device, timer)
        run_name = args.reducer+'_'+str(args.seed)
    elif args.reducer=='thresh':
        reducer = getattr(gradient_reducers, "ThreshReducer")(1, device, timer, thresh=args.thresh)
        run_name = args.reducer+'_'+str(args.thresh)+'_'+str(args.seed)
    elif args.reducer=='topk':
        reducer = getattr(gradient_reducers, "TopKReducer")(1, device, timer, compression=args.comp_ratio)
        run_name = args.reducer+'_'+str(args.comp_ratio)+'_'+str(args.seed)
    elif args.reducer=='gtopk':
        reducer = getattr(gradient_reducers, "GlobalTopKReducer")(1, device, timer, compression=args.comp_ratio)
        run_name = args.reducer+'_'+str(args.comp_ratio)+'_'+str(args.seed)
    elif args.reducer=='accordiontopk':
        reducer = getattr(gradient_reducers, "AccordionTopKReducer")(1, device, timer, k_low=args.k_low, k_high=args.k_high)
        run_name = 'acck' + '_' + str(args.k_low)+'_' + str(args.k_high) + '_' + str(args.seed)
    
    print("Reducer:",reducer)
    print ('Current cuda device after:', torch.cuda.current_device())
    
    # Wandb
    if args.use_wandb:
        if args.rank!=0:
            os.environ['WANDB_MODE'] = 'dryrun'
        config=vars(args)
        wandb.init(project="ncf-ml20m", name=run_name, entity=os.getenv('WANDB_ENTITY'),config=config)
    
    main_start_time = time.time()

    train_ratings = torch.load(args.data+'/train_ratings.pt', map_location=torch.device('cuda:{}'.format(args.local_rank)))
    test_ratings = torch.load(args.data+'/test_ratings.pt', map_location=torch.device('cuda:{}'.format(args.local_rank)))
    test_negs = torch.load(args.data+'/test_negatives.pt', map_location=torch.device('cuda:{}'.format(args.local_rank)))

    valid_negative = test_negs.shape[1]

    nb_maxs = torch.max(train_ratings, 0)[0]
    nb_users = nb_maxs[0].item() + 1
    nb_items = nb_maxs[1].item() + 1

    all_test_users = test_ratings.shape[0]

    test_users, test_items, dup_mask, real_indices = dataloading.create_test_data(test_ratings, test_negs, args)

    # make pytorch memory behavior more consistent later
    torch.cuda.empty_cache()

    # Create model
    model = NeuMF(nb_users, nb_items,
                  mf_dim=args.factors,
                  mlp_layer_sizes=args.layers,
                  dropout=args.dropout).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                          betas=(args.beta1, args.beta2), eps=args.eps)

    criterion = nn.BCEWithLogitsLoss(reduction='none').cuda() # use torch.mean() with dim later to avoid copy to host
    # Move model and loss to GPU

    if args.distributed:
        pass
        #model = DDP(model)

    local_batch = args.batch_size // args.world_size
    traced_criterion = torch.jit.trace(criterion.forward,
                                       (torch.rand(local_batch,1),torch.rand(local_batch,1)))

    print(model)
    print("{} parameters".format(utils.count_parameters(model)))

    if args.load_checkpoint_path:
        state_dict = torch.load(args.load_checkpoint_path)
        state_dict = {k.replace('module.', '') : v for k,v in state_dict.items()}
        model.load_state_dict(state_dict)

    if args.mode == 'test':
        start = time.time()
        hr, ndcg = val_epoch(model, test_users, test_items, dup_mask, real_indices, args.topk,
                             samples_per_user=valid_negative + 1,
                             num_user=all_test_users, distributed=args.distributed)
        val_time = time.time() - start
        eval_size = all_test_users * (valid_negative + 1)
        eval_throughput = eval_size / val_time

        print('best_eval_throughput:', eval_throughput)
        print('hr@10:', hr)
        return
    
    max_hr = 0
    best_epoch = 0
    train_throughputs, eval_throughputs = [], []
    
    #Compression related variables
    memories = [torch.zeros_like(param) for param in model.parameters() if param.requires_grad]
    send_buffers = [torch.zeros_like(param) for param in model.parameters() if param.requires_grad]
    names = [name for (name, _) in model.named_parameters()]
    
    total_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    bits_communicated=0
    total_params_transmitted=0
    for epoch in range(args.epochs):
        cur_bits = 0
        params_transmitted = 0
        begin = time.time()

        epoch_users, epoch_items, epoch_label = dataloading.prepare_epoch_train_data(train_ratings, nb_items, args)
        num_batches = len(epoch_users)
        for i in range(num_batches // args.grads_accumulated):
            for j in range(args.grads_accumulated):
                batch_idx = (args.grads_accumulated * i) + j
                user = epoch_users[batch_idx]
                item = epoch_items[batch_idx]
                label = epoch_label[batch_idx].view(-1,1)

                outputs = model(user, item)
                loss = traced_criterion(outputs, label).float()
                loss = torch.mean(loss.view(-1), 0)
                loss.backward()
                
            grads = [param.grad.data.clone().detach() for param in model.parameters()]
            for grad, memory, send_bfr in zip(grads, memories, send_buffers):
                send_bfr.data[:] = grad + memory
                
            cur_bits, cur_params = reducer.reduce(send_buffers, grads, memories)
            bits_communicated += cur_bits
            params_transmitted += cur_params
            
            for param, grad in zip(model.parameters(), grads):
                param.grad.data[:] = grad
        
            optimizer.step()
            for param, grad in zip(model.parameters(), grads):
                param.grad = None
        grad_norm_sq = 0
        grad_norm = 0
        for grad in grads:
            grad_norm_sq += l2norm(grad)**2
        grad_norm = torch.sqrt(grad_norm_sq).item()
        total_params_transmitted += params_transmitted
        ratio_transmitted = params_transmitted/(num_batches*total_params)*100
        avg_ratio_transmitted = total_params_transmitted/(num_batches*total_params*(epoch+1.0))*100
        if args.use_wandb:
            wandb.log({'grad_norm'+'/'+'entire_model': grad_norm}, step=int(epoch+1))
            wandb.log({'density'+'/'+'current': ratio_transmitted}, step=int(epoch+1))
            wandb.log({'density'+'/'+'average': avg_ratio_transmitted}, step=int(epoch+1))
            wandb.log({'density'+'/'+'bits': bits_communicated}, step=int(epoch+1))
            wandb.log({'density'+'/'+'current': ratio_transmitted}, step=int(epoch+1))
        print("###### Epoch stats: #######")
        print("Current density:", ratio_transmitted)
        print("Average Density:", avg_ratio_transmitted)
        print("Total bits transmitted:", bits_communicated)
        
        del epoch_users, epoch_items, epoch_label
        train_time = time.time() - begin
        begin = time.time()

        epoch_samples = len(train_ratings) * (args.negative_samples + 1)
        train_throughput = epoch_samples / train_time
        train_throughputs.append(train_throughput)

        hr, ndcg = val_epoch(model, test_users, test_items, dup_mask, real_indices, args.topk,
                             samples_per_user=valid_negative + 1,
                             num_user=all_test_users, epoch=epoch, distributed=args.distributed)

        val_time = time.time() - begin


        eval_size = all_test_users * (valid_negative + 1)
        eval_throughput = eval_size / val_time
        eval_throughputs.append(eval_throughput)
        cur_stats = {'train_throughput': train_throughput,
                              'hr@10': hr,
                              'train_epoch_time': train_time,
                              'validation_epoch_time': val_time,
                              'eval_throughput': eval_throughput}
        epoch_stats = {'epoch_stats'+'/'+ key : value for key, value in cur_stats.items()}
        wandb.log(epoch_stats, step=int(epoch+1))

        if hr > max_hr and args.rank == 0:
            max_hr = hr
            best_epoch = epoch
#            save_checkpoint_path = os.path.join(args.checkpoint_dir, 'model.pth')
            print("New best hr!")
#            torch.save(model.state_dict(), save_checkpoint_path)
            best_model_timestamp = time.time()
            best_stats={'best_train_throughput': max(train_throughputs),
                'best_eval_throughput': max(eval_throughputs),
                'mean_train_throughput': np.mean(train_throughputs),
                'mean_eval_throughput': np.mean(eval_throughputs),
                'best_accuracy': max_hr,
                'best_epoch': best_epoch,
                'time_to_target': time.time() - main_start_time,
                'time_to_best_model': best_model_timestamp - main_start_time}
            best_epoch_stats = {'best_epoch_stats'+'/'+key : value for key, value in best_stats.items()}
            print(best_epoch_stats)
            wandb.log(best_epoch_stats, step=int(best_epoch+1))

        if args.threshold is not None:
            if hr >= args.threshold:
                print("Hit threshold of {}".format(args.threshold))
                break
       
        # Log relative compression error and memory norm
        model_memory_sq_norm = 0.0
        model_send_bfr_sq_norm = 0.0
        model_memory_inf_norm = 0.0
        error_dict = {}
        memory_norm = {}
        memory_inf_norm = {}
        for name, memory, send_bfr in zip(names, memories, send_buffers):
            memory_sq_norm = l2norm(memory)**2
            send_bfr_sq_norm = l2norm(send_bfr)**2
            rel_compression_error = memory_sq_norm/send_bfr_sq_norm
            
            model_memory_sq_norm += memory_sq_norm
            model_send_bfr_sq_norm += send_bfr_sq_norm
            
            error_dict[name] = rel_compression_error
            memory_norm[name] = torch.sqrt(memory_sq_norm).item()
            
            memory_inf_norm[name] = torch.max(memory.abs()).item()
            model_memory_inf_norm = max(memory_inf_norm[name], model_memory_inf_norm)
            
        model_rel_compression_error = model_memory_sq_norm/model_send_bfr_sq_norm
        model_memory_norm = torch.sqrt(model_memory_sq_norm).item()
        
        error_dict["entire_model"] = model_rel_compression_error
        memory_norm["entire_model"] = model_memory_norm
        memory_inf_norm["entire_model"] = model_memory_inf_norm
        
        error_dict = {'rel_compression_error'+'/'+ key : value for key, value in error_dict.items()}
        memory_norm = {'memory_norm'+'/'+key: value for key, value in memory_norm.items()}
        memory_inf_norm = {'memory_inf_norm'+'/'+key: value for key, value in memory_norm.items()}
        
        if args.use_wandb:
            wandb.log(error_dict, step=int(epoch+1))
            wandb.log(memory_norm, step=int(epoch+1))
            wandb.log(memory_inf_norm, step=int(epoch+1))
        
            
@torch.jit.script
def l2norm(tensor):
    """Compute the L2 Norm of a tensor in a fast and correct way"""
    return torch.sqrt(torch.sum(tensor ** 2))


if __name__ == '__main__':
    main()