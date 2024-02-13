import sys
import os
import argparse
import build_network
from torchvision import transforms
import pickle
from robustness.datasets import ImageNet

from robustness import train
from cox.utils import Parameters
from cox import store

from robustness import model_utils, datasets, train, defaults
from robustness.data_augmentation import RandomNoise

import torch as ch 
import numpy as np

def convert_string_frac_to_float(s):
    s = s.split('/')
    if len(s) == 2:
        return float(s[0]) / float(s[1])
    elif len(s) == 1:
        return float(s[0])
    else:
        raise ValueError('Cannot process this fraction')

parser = argparse.ArgumentParser(description='Input parameters for evaluating model on adversarial robustness')
parser.add_argument('-R', '--RANDOMSEED', metavar='--R', type=int, default=0, help='random seed to use for synthesis')
parser.add_argument('-N', '--NUMSAMPLES', metavar='--N', type=int, default=16, help='number of samples to evaluate from the test data')
parser.add_argument('-I', '--MAXITER', metavar='--I', type=int, default=32, help='maximum number of iterations to generate adversarial examples')
parser.add_argument('-B', '--BATCHSIZE', metavar='--B', type=int, default=16, help='batch size for evaluation, set based on GPU limitations')
parser.add_argument('-T', '--ATTACK_TYPE', metavar='--T', type=str, default='inf', help='norm of the adversarial perturbation (“inf”, 1 or 2.)')
parser.add_argument('-D', '--EPS_DIVIDE', metavar='--D', type=int, default=5, help='divide the eps by this to set the step size')
parser.add_argument('-L', '--MIN_EPS', metavar='--L', type=int, default=-4, help='lowest eps value to test')
parser.add_argument('-M', '--MAX_EPS', metavar='--M', type=int, default=-1, help='maximum eps value to test')
parser.add_argument('-U', '--NUM_EPS_STEPS_BETWEEN', metavar='--U', type=int, default=2, help='number of eps steps between each power of 10')
parser.add_argument('-O', '--OVERWRITE_PICKLE', metavar='--P', type=bool, default=False, help='set to true to overwrite the saved pckl file, if false then exits out if the file already exists')
parser.add_argument('-E', '--ATTACK_EPS_LIST', metavar='--E', type=str, help='if given, use this as the list of attacks rather than the range between min and max eps. Separate by commas.')

args=parser.parse_args()
if args.ATTACK_EPS_LIST is not None:
    print(args.ATTACK_EPS_LIST)
    print('Using list of attack eps %s, not min/max default'%args.ATTACK_EPS_LIST)
    eps_list = [convert_string_frac_to_float(e) for e in args.ATTACK_EPS_LIST.split(',')]
    print(eps_list)
else:
    num_eps_steps = int((args.MAX_EPS - args.MIN_EPS) * args.NUM_EPS_STEPS_BETWEEN + 1)
    eps_list = np.concatenate([[0], np.logspace(args.MIN_EPS, args.MAX_EPS, num_eps_steps)])
RANDOMSEED = args.RANDOMSEED
NUMSAMPLES=args.NUMSAMPLES
MAXITER=args.MAXITER
BATCHSIZE=args.BATCHSIZE
EPS_DIVIDE=args.EPS_DIVIDE
attack_norm=args.ATTACK_TYPE

NUM_WORKERS=4

if args.ATTACK_EPS_LIST is None:
    # Make a directory for saving:
    base_filepath = 'adversarial_eval_robustness_lib/rs%d_nsamp%d_maxiter%d_type%s_epsdiv%d_steps%d'%(
                        RANDOMSEED, NUMSAMPLES, MAXITER, str(attack_norm), EPS_DIVIDE, args.NUM_EPS_STEPS_BETWEEN)
else:
    # Make a directory for saving:
    base_filepath = 'adversarial_eval_robustness_lib/rs%d_nsamp%d_maxiter%d_type%s_epsdiv%d_eps%s'%(
                        RANDOMSEED, NUMSAMPLES, MAXITER, str(attack_norm), EPS_DIVIDE, str(args.ATTACK_EPS_LIST).replace('/', '|'))
print(base_filepath)
try:
    os.makedirs(base_filepath)
except:
    pass
pckl_path = base_filepath + '/adv_eval.pckl'

if os.path.isfile(pckl_path) and not args.OVERWRITE_PICKLE:
    raise FileExistsError('The file %s already exists, and you are not forcing overwriting'%pckl_path)

# Run model on range of EPS attacks
save_accuracy_eps = []

# Build the model
model, ds = build_network.main()

ch.manual_seed(RANDOMSEED)
np.random.seed(RANDOMSEED)

# All of the models tested here had the same size.  
# If running a different model this might need to be modified. 
TEST_TRANSFORMS_IMAGENET = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

ds.transform_test = TEST_TRANSFORMS_IMAGENET
ds.transform_train = TEST_TRANSFORMS_IMAGENET

_, val_loader = ds.make_loaders(batch_size=BATCHSIZE, 
                                           workers=NUM_WORKERS, 
                                           shuffle_train=False, 
                                           shuffle_val=True,
                                           data_aug=True,
                                           only_val=True,
                                           subset_val=NUMSAMPLES, 
                                           seed=RANDOMSEED,
                                          )

for eps in eps_list:
    print('Evaluating EPS %s, NORM %s'%(eps, attack_norm))
    # Hard-coded base parameters
    eval_args = {
        'out_dir': "eval_out",
        'exp_name': "eval_natural_imagenet",
        'adv_train': 0,
        "adv_eval":1, 
        'constraint': attack_norm,
        'eps': eps,
        'step_size': eps/EPS_DIVIDE,
        'attack_lr': eps/EPS_DIVIDE,
        'attack_steps': MAXITER,
        'save_ckpt_iters':1,
        'out_dir':'adv_eval_robustness',
        'targeted':False,
    }
    
    eval_args = Parameters(eval_args)
    
    # Fill whatever parameters are missing from the defaults
    eval_args = defaults.check_and_fill_args(eval_args,
                            defaults.TRAINING_ARGS, ImageNet)
    eval_args = defaults.check_and_fill_args(eval_args,
                            defaults.PGD_ARGS, ImageNet)
    
    log_info = train.eval_model(eval_args, model, val_loader,store=None, adv_only=True)
    save_accuracy_eps.append(log_info['adv_prec1'].detach().cpu())

pckl_output_dict = {'RANDOMSEED':RANDOMSEED,
                    'NUMSAMPLES':NUMSAMPLES,
                    'BATCHSIZE':BATCHSIZE,
                    'MAXITER':MAXITER,
                    'EPS_DIVIDE':EPS_DIVIDE,
                    'eps_list':eps_list,
                    'save_accuracy_eps':save_accuracy_eps,
                    }

with open(pckl_path, 'wb') as handle:
    pickle.dump(pckl_output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
