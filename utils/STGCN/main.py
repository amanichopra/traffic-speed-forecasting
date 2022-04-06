# @Time     : Jan. 02, 2019 22:17
# @Author   : Veritas YIN
# @FileName : main.py
# @Version  : 1.0
# @Project  : Orion
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from os.path import join as pjoin

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

from utils.math_graph import *
from utils.data_utils import *
from models.trainer import model_train
from models.tester import model_test, save_results

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_route', type=int, default=105)
parser.add_argument('--n_his', type=int, default=12)
parser.add_argument('--n_pred', type=int, default=9)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--ks', type=int, default=3) # kernel size
parser.add_argument('--inf_mode', type=str, default='merge')
parser.add_argument('--speeds_path', type=str, default='../../data/processed/fwy_405_n_ds/speeds_form.csv')
parser.add_argument('--adj_mat_path', type=str, default='../../data/processed/fwy_405_n_ds/adj_mat_form.csv')
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--save', type=int, default=10) # num epochs to save for
parser.add_argument('--kt', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--opt', type=str, default='RMSProp')

print()
print('==> Running main.py...')
print()

args = parser.parse_args()
print(f'Training configs: {args}')

n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
Ks = args.ks
# blocks: settings of channel size in st_conv_blocks / bottleneck design
blocks = [[1, 32, 64], [64, 32, 128]]

# Load wighted adjacency matrix W
speeds_path, adj_mat_path = args.speeds_path, args.adj_mat_path
W = weight_matrix(adj_mat_path)

# Calculate graph kernel
L = scaled_laplacian(W)
# Alternative approximation method: 1st approx - first_approx(W, n).
Lk = cheb_poly_approx(L, Ks, n)
tf.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float32))

# Data Preprocessing
n_train, n_val, n_test = 34, 5, 5
PeMS = data_gen(speeds_path, (n_train, n_val, n_test), n, n_his + n_pred)
print(f'>> Loading dataset with Mean: {PeMS.mean:.2f}, STD: {PeMS.std:.2f}')

if __name__ == '__main__':
    model_train(PeMS, blocks, args)
    model_test(PeMS, PeMS.get_len('test'), n_his, n_pred, args.inf_mode)

    save_results(PeMS, PeMS.get_len('test'), n_his, n_pred, args.inf_mode, '../../models/trained/STGCN/preds', '../../models/trained/STGCN')

