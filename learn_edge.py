"""Unified interface to all dynamic graph model experiments"""
import math
import logging
import time
import random
import sys
import argparse

import torch
import pandas as pd
import numpy as np
#import numba
from tqdm import tqdm

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing

from module import TGAN
from graph import NeighborFinder
from utils import EarlyStopMonitor, RandEdgeSampler

### Argument and global variables
parser = argparse.ArgumentParser('Interface for TGAT experiments on link predictions')
parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='u2k_i200_1W')
parser.add_argument('--bs', type=int, default=200, help='batch_size')
parser.add_argument('--prefix', type=str, default='', help='prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimentions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimentions of the time embedding')
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method', default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod', help='use dot product attention or mapping based')
parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], help='how to use time information', default='time')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
parser.add_argument('--scale_label', type=str, choices=['none', 'MinMax', 'Log', 'Cbrt', 'Quantile'], default='Cbrt', help='how to scale the label')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim


MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}-{epoch}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)



def eval_one_epoch(hint, tgan, sampler, src, dst, ts, label, label_raw):
    val_mae_raw, val_mae, val_r2_raw, val_r2 = [], [], [], []
    with torch.no_grad():
        tgan = tgan.eval()
        TEST_BATCH_SIZE=30
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        for k in range(num_test_batch):
            # percent = 100 * k / num_test_batch
            # if k % int(0.2 * num_test_batch) == 0:
            #     logger.info('{0} progress: {1:10.4f}'.format(hint, percent))
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance - 1, s_idx + TEST_BATCH_SIZE)
            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]
            # label_l_cut = label[s_idx:e_idx]
            
            size = len(src_l_cut)
            _, dst_l_fake = sampler.sample(size)
            pos_label = label[s_idx:e_idx]
            pos_label_raw = label_raw[s_idx:e_idx]
            neg_label = np.zeros(size)
            pos_pred, neg_pred = tgan.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)
            
            pred_score = np.concatenate([(pos_pred).cpu().numpy(), (neg_pred).cpu().numpy()])
            # pred_label = pred_score > 0.5
            true_label = np.concatenate([pos_label, neg_label])
            true_label_raw = np.concatenate([pos_label_raw, neg_label])
            val_mae.append(mean_absolute_error(true_label, pred_score))
            val_r2.append(r2_score(true_label, pred_score))
            if args.scale_label != 'none':
                pred_score_raw = convert_to_raw_label_scale(np.concatenate([dst_l_cut,dst_l_fake]), pred_score)
                val_r2_raw.append(r2_score(true_label_raw, pred_score_raw))
                val_mae_raw.append(mean_absolute_error(true_label_raw, pred_score_raw))

    return np.mean(val_mae_raw), np.mean(val_mae), np.mean(val_r2_raw), np.mean(val_r2)

### Load data and train val test split
g_df = pd.read_csv('./processed/ml_{}.csv'.format(DATA))
# g_df_raw = g_df.copy()
e_feat = np.load('./processed/ml_{}.npy'.format(DATA))
n_feat = np.load('./processed/ml_{}_node.npy'.format(DATA))

# e_feat = np.zeros((len(e_feat), 2))
# n_feat = np.zeros((len(n_feat), 2))

val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))

ts_l = g_df.ts.values
valid_train_flag = (ts_l <= val_time)


train_df = g_df[valid_train_flag]
train_i = train_df.i.unique()
# filter out new merchants in the valid/test
from scipy import stats

g_df = g_df[g_df.i.isin(train_i)]

ts_l = g_df.ts.values
valid_train_flag = (ts_l <= val_time)
src_l = g_df.u.values
dst_l = g_df.i.values
cat_l = g_df.cat.values
e_idx_l = g_df.idx.values

max_src_index = src_l.max()
max_idx = max(src_l.max(), dst_l.max())

random.seed(2020)

total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))
num_total_unique_nodes = len(total_node_set)

# scaling labels
g_df['raw_label'] = g_df.label.copy()
raw_label_l = g_df.raw_label.values
# NOTE!!!!!!! change TGAN.contrast if the label range is changed
i2cat = g_df.groupby('i').first().reset_index().set_index('i')['cat'].to_dict()
for cat in i2cat.values():
    orig_labels = g_df.loc[(g_df.cat == cat) & valid_train_flag, 'label'].values
    lower = np.quantile(orig_labels, 0.001)
    upper = np.quantile(orig_labels, 0.999)
    g_df.loc[(g_df.cat == cat) & valid_train_flag, 'label'] = np.clip(orig_labels, lower, upper)
if args.scale_label == 'none':
    label_l = g_df.label.values
else:
    # train_df['abs_label'] = train_df['label'].abs()
    
    # i_maxes = train_df.groupby('i')['abs_label'].max().reset_index().set_index('i')['abs_label'].to_dict()
    # # normalize labels with the max value from training set
    # for i, max_label in i_maxes.items():
    #     g_df.loc[g_df.i == i, 'label'] /= max_label
    # label_l = g_df.label.values
    # def convert_to_raw_label_scale(dst_l_cut, preds):
    #     if isinstance(preds, np.ndarray):
    #         scale = np.array([i_maxes[dst] for dst in dst_l_cut])
    #     else:
    #         scale = torch.tensor([i_maxes[dst] for dst in dst_l_cut], dtype=float, device=device)
    #     return preds * scale, labels * scale
    train_df = g_df[valid_train_flag]
    if args.scale_label == 'MinMax':
        scaler = preprocessing.MinMaxScaler
    elif args.scale_label == 'Quantile':
        scaler = preprocessing.QuantileTransformer
    elif args.scale_label == 'Log':
        scaler = preprocessing.StandardScaler
    elif args.scale_label == 'Cbrt':
        scaler = preprocessing.StandardScaler
    
    def init_transform(label_vals):
        if args.scale_label == 'Log':
            label_vals = np.sign(label_vals) * np.log(np.abs(label_vals)+1)
        elif args.scale_label == 'Cbrt':
            label_vals = np.cbrt(label_vals)
        return label_vals

    scalers = {}
    if args.scale_label == 'Cbrt':
        g_df.label = np.cbrt(g_df.label.values)
    else:
        for cat in i2cat.values():
            cat_train_df = train_df[train_df.cat==cat]
            scalers[cat] = scaler()
            train_label_vals = init_transform(cat_train_df.label.values)
            scalers[cat].fit(train_label_vals.reshape(-1, 1))
            label_vals = init_transform(g_df.loc[g_df.cat == cat]['label'].values)        
            g_df.loc[g_df.cat == cat, 'label'] = scalers[cat].transform(label_vals.reshape(-1, 1))
    
    def convert_to_raw_label_scale(dst_l_cut, preds):
        raw_preds = []
        for dst, pred in zip(dst_l_cut, preds):
            cat = i2cat[dst]
            if args.scale_label == 'Cbrt':
                raw = np.power(pred, 3)
                raw_preds.append(raw)
            else:
                raw = scalers[cat].inverse_transform(pred.reshape(1, -1))
                if args.scale_label == 'Log':
                    raw = np.sign(raw) * (np.exp(np.abs(raw))-1)
                elif args.scale_label == 'Cbrt':
                    raw = np.power(raw, 3)
                raw_preds.append(raw[0, 0])
        if isinstance(preds, np.ndarray):
            return np.array(raw_preds)
        else:
            return torch.tensor(raw_preds, dtype=float, device=device)
    label_l = g_df.label.values



train_src_l = src_l[valid_train_flag]
train_dst_l = dst_l[valid_train_flag]
train_ts_l = ts_l[valid_train_flag]
train_e_idx_l = e_idx_l[valid_train_flag]
train_label_l = label_l[valid_train_flag]
train_raw_label_l = raw_label_l[valid_train_flag]
train_cat_l = cat_l[valid_train_flag]

# define the new nodes sets for testing inductiveness of the model
train_node_set = set(train_src_l).union(train_dst_l)
# assert(len(train_node_set - mask_node_set) == len(train_node_set))
new_node_set = total_node_set - train_node_set

# select validation and test dataset
valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
valid_test_flag = ts_l > test_time

# is_new_node_edge = np.array([(a in new_node_set or b in new_node_set) for a, b in zip(src_l, dst_l)])
# nn_val_flag = valid_val_flag * is_new_node_edge
# nn_test_flag = valid_test_flag * is_new_node_edge

# validation and test with all edges
val_src_l = src_l[valid_val_flag]
val_dst_l = dst_l[valid_val_flag]
val_ts_l = ts_l[valid_val_flag]
val_e_idx_l = e_idx_l[valid_val_flag]
val_label_l = label_l[valid_val_flag]
val_raw_label_l = raw_label_l[valid_val_flag]
val_cat_l = cat_l[valid_val_flag]

test_src_l = src_l[valid_test_flag]
test_dst_l = dst_l[valid_test_flag]
test_ts_l = ts_l[valid_test_flag]
test_e_idx_l = e_idx_l[valid_test_flag]
test_label_l = label_l[valid_test_flag]
test_raw_label_l = raw_label_l[valid_test_flag]
test_cat_l = cat_l[valid_test_flag]

# # validation and test with edges that at least has one new node (not in training set)
# nn_val_src_l = src_l[nn_val_flag]
# nn_val_dst_l = dst_l[nn_val_flag]
# nn_val_ts_l = ts_l[nn_val_flag]
# nn_val_e_idx_l = e_idx_l[nn_val_flag]
# nn_val_label_l = label_l[nn_val_flag]

# nn_test_src_l = src_l[nn_test_flag]
# nn_test_dst_l = dst_l[nn_test_flag]
# nn_test_ts_l = ts_l[nn_test_flag]
# nn_test_e_idx_l = e_idx_l[nn_test_flag]
# nn_test_label_l = label_l[nn_test_flag]

### Initialize the data structure for graph and edge sampling
# build the graph for fast query
# graph only contains the training data (with 10% nodes removal)
adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
    adj_list[src].append((dst, eidx, ts))
    adj_list[dst].append((src, eidx, ts))
train_ngh_finder = NeighborFinder(adj_list, uniform=UNIFORM)

# full graph with all the data for the test and validation purpose
full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
full_ngh_finder = NeighborFinder(full_adj_list, uniform=UNIFORM)

train_rand_sampler = RandEdgeSampler(train_src_l, train_dst_l)
val_rand_sampler = RandEdgeSampler(src_l, dst_l)
# nn_val_rand_sampler = RandEdgeSampler(nn_val_src_l, nn_val_dst_l)
test_rand_sampler = RandEdgeSampler(src_l, dst_l)
# nn_test_rand_sampler = RandEdgeSampler(nn_test_src_l, nn_test_dst_l)


### Model initialize
# device = torch.device('cuda:{}'.format(GPU))
device = torch.device('cpu')
tgan = TGAN(train_ngh_finder, n_feat, e_feat,
            num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
            seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT, node_dim=NODE_DIM, time_dim=TIME_DIM)
optimizer = torch.optim.Adam(tgan.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.MSELoss()
tgan = tgan.to(device)

num_instance = len(train_src_l)
num_batch = math.ceil(num_instance / BATCH_SIZE)

logger.info('num of training instances: {}'.format(num_instance))
logger.info('num of batches per epoch: {}'.format(num_batch))
idx_list = np.arange(num_instance)
np.random.shuffle(idx_list) 

early_stopper = EarlyStopMonitor(higher_better=False)
for epoch in range(NUM_EPOCH):
    # Training 
    # training use only training graph
    tgan.ngh_finder = train_ngh_finder
    mae_raw, mae, r2_raw, r2, m_loss = [], [], [], [], []
    np.random.shuffle(idx_list)
    logger.info('start {} epoch'.format(epoch))
    for k in tqdm(range(num_batch)):
        # percent = 100 * k / num_batch
        # if k % int(0.2 * num_batch) == 0:
        #     logger.info('progress: {0:10.4f}'.format(percent))

        s_idx = k * BATCH_SIZE
        e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
        src_l_cut, dst_l_cut = train_src_l[s_idx:e_idx], train_dst_l[s_idx:e_idx]
        ts_l_cut = train_ts_l[s_idx:e_idx]
        label_l_cut = train_label_l[s_idx:e_idx]
        raw_label_l_cut = train_raw_label_l[s_idx:e_idx]
        size = len(src_l_cut)
        _, dst_l_fake = train_rand_sampler.sample(size)
        
        with torch.no_grad():
            pos_label = torch.tensor(label_l_cut, dtype=torch.float, device=device)
            neg_label = torch.zeros(size, dtype=torch.float, device=device)
        
        optimizer.zero_grad()
        tgan = tgan.train()
        pos_pred, neg_pred = tgan.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)

        # if args.scale_label == 'none':
        loss = criterion(pos_pred, pos_label)
        loss += criterion(neg_pred, neg_label)
        # else:
        #     pos_pred_raw, pos_label_raw = convert_to_raw_label_scale(dst_l_cut, pos_pred, pos_label)
        #     neg_pred_raw, neg_label_raw = convert_to_raw_label_scale(dst_l_fake, neg_pred, neg_label)
        #     loss = criterion(pos_pred_raw, pos_label_raw)
        #     loss += criterion(neg_pred_raw, neg_label_raw)
        
        loss.backward()
        optimizer.step()
        # get training results
        with torch.no_grad():
            tgan = tgan.eval()
            pred_score = np.concatenate([(pos_pred).cpu().detach().numpy(), (neg_pred).cpu().detach().numpy()])
            true_label = np.concatenate([label_l_cut, np.zeros(size)])
            true_label_raw = np.concatenate([raw_label_l_cut, np.zeros(size)])
            
            mae.append(mean_absolute_error(true_label, pred_score))

            # f1.append(f1_score(true_label, pred_label))
            m_loss.append(loss.item())
            r2.append(r2_score(true_label, pred_score))
            if args.scale_label != 'none':
                pred_score_raw = convert_to_raw_label_scale(np.concatenate([dst_l_cut,dst_l_fake]), pred_score)
                mae_raw.append(mean_absolute_error(true_label_raw, pred_score_raw))
                r2_raw.append(r2_score(true_label_raw, pred_score_raw))

    # validation phase use all information
    tgan.ngh_finder = full_ngh_finder
    val_mae_raw, val_mae, val_r2_raw, val_r2 = eval_one_epoch('val for old nodes', tgan, val_rand_sampler, val_src_l, 
    val_dst_l, val_ts_l, val_label_l, val_raw_label_l)

    # nn_val_acc, nn_val_ap, nn_val_f1, nn_val_auc = eval_one_epoch('val for new nodes', tgan, val_rand_sampler, nn_val_src_l, 
    # nn_val_dst_l, nn_val_ts_l, nn_val_label_l)
        
    logger.info('epoch: {}:'.format(epoch))
    logger.info('Epoch mean loss: {:.4f}'.format(np.mean(m_loss)))
    # logger.info('train acc: {}, val acc: {}, new node val acc: {}'.format(np.mean(acc), val_acc, nn_val_acc))
    # logger.info('train R2: {}, val R2: {}, new node val R2: {}'.format(np.mean(r2), val_auc, nn_val_auc))
    # logger.info('train MAE: {}, val ap: {}, new node val MAE: {}'.format(np.mean(mae), val_ap, nn_val_ap))
    # logger.info('train f1: {}, val f1: {}, new node val f1: {}'.format(np.mean(f1), val_f1, nn_val_f1))

    logger.info('train R2: {:.4f}, val R2: {:.4f}'.format(np.mean(r2), val_r2))
    logger.info('train MAE: {:.4f}, val MAE: {:.4f}'.format(np.mean(mae), val_mae))
    if args.scale_label != 'none':
        logger.info('train raw R2: {:.4f}, val raw R2: {:.4f}'.format(np.mean(r2_raw), val_r2_raw))
        logger.info('train raw MAE: {:.4f}, val raw MAE: {:.4f}'.format(np.mean(mae_raw), val_mae_raw))

    if early_stopper.early_stop_check(val_mae):
        logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
        logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
        best_model_path = get_checkpoint_path(early_stopper.best_epoch)
        tgan.load_state_dict(torch.load(best_model_path))
        logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
        tgan.eval()
        break
    else:
        torch.save(tgan.state_dict(), get_checkpoint_path(epoch))


# testing phase use all information
tgan.ngh_finder = full_ngh_finder
test_acc, test_ap, test_f1, test_auc = eval_one_epoch('test for old nodes', tgan, test_rand_sampler, test_src_l, 
test_dst_l, test_ts_l, test_label_l, test_raw_label_l)

# nn_test_acc, nn_test_ap, nn_test_f1, nn_test_auc = eval_one_epoch('test for new nodes', tgan, nn_test_rand_sampler, nn_test_src_l, 
# nn_test_dst_l, nn_test_ts_l, nn_test_label_l)

logger.info('Test statistics: Old nodes -- acc: {}, auc: {}, ap: {}'.format(test_acc, test_auc, test_ap))
# logger.info('Test statistics: New nodes -- acc: {}, auc: {}, ap: {}'.format(nn_test_acc, nn_test_auc, nn_test_ap))

logger.info('Saving TGAN model')
torch.save(tgan.state_dict(), MODEL_SAVE_PATH)
logger.info('TGAN models saved')

 




