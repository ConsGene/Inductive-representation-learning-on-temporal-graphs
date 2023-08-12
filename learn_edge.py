"""Unified interface to all dynamic graph model experiments"""
import math
import logging
import time
import random
import sys
import argparse

import torch
import numpy as np
# import numba
from tqdm import tqdm

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from data_utils import get_data

from module import TGAN
from graph import NeighborFinder
from utils import EarlyStopMonitor, RandEdgeSampler
import os

# Argument and global variables
parser = argparse.ArgumentParser(
    'Interface for TGAT experiments on link predictions')
parser.add_argument('-d', '--data', type=str,
                    help='data sources to use, try wikipedia or reddit', default='u2k_i200_1W')
parser.add_argument('--bs', type=int, default=200, help='batch_size')
parser.add_argument('--prefix', type=str, default='',
                    help='prefix to name the checkpoints')
parser.add_argument('--loss', type=str,
                    choices=['l1', 'l2','cross_entropy'], default='cross_entropy', help='type of loss')
parser.add_argument('--n_degree', type=int, default=20,
                    help='number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2,
                    help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2,
                    help='number of network layers')

parser.add_argument('--n_classes', type=int, default=10,
                    help='number of classes')

parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.1,
                    help='dropout probability')
parser.add_argument('--gpu', type=int, default=0,
                    help='idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100,
                    help='Dimentions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100,
                    help='Dimentions of the time embedding')
parser.add_argument('--agg_method', type=str, choices=[
    'attn', 'lstm', 'mean'], help='local aggregation method', default='attn')
parser.add_argument('--attn_mode', type=str, choices=[
    'prod', 'map'], default='prod', help='use dot product attention or mapping based')
parser.add_argument('--time', type=str, choices=[
    'time', 'pos', 'empty'], help='how to use time information', default='time')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--scale_label', type=str, choices=[
    'none', 'MinMax', 'Log', 'Cbrt', 'Quantile','Discr'], default='Discr', help='how to scale the label')

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
NUM_CLASSES = args.n_classes
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
SCALE = args.scale_label

MODEL_NAME = f'{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}'
MODEL_SAVE_PATH = f'./saved_models/{MODEL_NAME}.pth'
BEST_METRICS_PATH = f'./best_metrics/{args.data}-best-metrics.txt'


def get_checkpoint_path(
        epoch): return f'./saved_checkpoints/{MODEL_NAME}-{epoch}.pth'


# set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

# if GPU >= 0:
# device = torch.device('cuda:{}'.format(GPU))
# else:


if GPU >= 0:
    device = torch.device('cuda:{}'.format(GPU))
else:
    device = torch.device('cpu')


# device = torch.device('cpu')
# node_features, edge_features, full_data, train_data, val_data, test_data, scaleUtil = get_data(
#     DATA, SCALE, device)


# write a function to store the best metrics and the corresponding model checkpoint


def store_checkpoint(epoch, tgan, optimizer):
    latest_path = get_checkpoint_path('latest')
    torch.save({
        'epoch': epoch,
        'model_state_dict': tgan.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, latest_path)
    logger.info('Checkpoint saved to {}'.format(latest_path))
    return


def eval_one_epoch(tgan, sampler, src, dst, ts, label, label_raw):
    val_mae_raw, val_mae, val_r2_raw, val_r2 = [], [], [], []
    with torch.no_grad():
        tgan = tgan.eval()
        TEST_BATCH_SIZE = 30
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
            pos_pred, neg_pred = tgan.contrast(
                src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)

            pred_score = np.concatenate(
                [(pos_pred).cpu().numpy(), (neg_pred).cpu().numpy()])
            # pred_label = pred_score > 0.5
            true_label = np.concatenate([pos_label, neg_label])
            true_label_raw = np.concatenate([pos_label_raw, neg_label])
            val_mae.append(mean_absolute_error(true_label, pred_score))
            val_r2.append(r2_score(true_label, pred_score))
            if args.scale_label != 'none':
                pred_score_raw = scaleUtil.convert_to_raw_label_scale(
                    np.concatenate([dst_l_cut, dst_l_fake]), pred_score)
                val_r2_raw.append(r2_score(true_label_raw, pred_score_raw))
                val_mae_raw.append(mean_absolute_error(
                    true_label_raw, pred_score_raw))

    return np.mean(val_mae_raw), np.mean(val_mae), np.mean(val_r2_raw), np.mean(val_r2)

def eval_one_epoch_discr(tgan, sampler, src, dst, ts, label, label_raw):
    val_mae_raw, val_mae, val_r2_raw, val_r2 = [], [], [], []
    with torch.no_grad():
        tgan = tgan.eval()
        TEST_BATCH_SIZE = 30
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        for k in range(num_test_batch):
            
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance - 1, s_idx + TEST_BATCH_SIZE)
            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]
            size = len(src_l_cut)
            _, dst_l_fake = sampler.sample(size)
            
            pos_label = label[s_idx:e_idx]
            pos_label_raw = label_raw[s_idx:e_idx]
            neg_label = np.zeros(size)
            
            pos_pred, neg_pred = tgan.contrast(
                src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)

            pred_score_discrete = np.concatenate(
                [(pos_pred).cpu().detach().numpy(), (neg_pred).cpu().detach().numpy()])
            pred_class = np.argmax(pred_score_discrete, axis=1)
            pred_score = get_centroid_preds(np.concatenate([dst_l_cut, dst_l_fake]), pred_class)
            
            true_discrete_labels = np.concatenate([pos_label, np.zeros((size, num_classes))])
            true_labels_class = np.argmax(true_discrete_labels, axis=1)
            true_label = get_centroid_preds(np.concatenate([dst_l_cut, dst_l_fake]), true_labels_class)
            
            true_label_raw = np.concatenate([pos_label_raw, neg_label])
            
            val_mae.append(mean_absolute_error(true_label.cpu().numpy(), pred_score.cpu().numpy()))
            val_r2.append(r2_score(true_label.cpu().numpy(), pred_score.cpu().numpy()))
            if scale_label != 'none':
                val_mae_raw.append(mean_absolute_error(
                    true_label_raw, pred_score.cpu().numpy()))
                val_r2_raw.append(r2_score(true_label_raw, pred_score.cpu().numpy()))
    

    return np.mean(val_mae_raw), np.mean(val_mae), np.mean(val_r2_raw), np.mean(val_r2)    


max_src_index = full_data.sources.max()
max_idx = max(full_data.sources.max(), full_data.destinations.max())
# Initialize the data structure for graph and edge sampling
# build the graph for fast query
# graph only contains the training data (with 10% nodes removal)
adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(train_data.sources, train_data.destinations, train_data.edge_idxs, train_data.timestamps):
    adj_list[src].append((dst, eidx, ts))
    adj_list[dst].append((src, eidx, ts))
train_ngh_finder = NeighborFinder(adj_list, uniform=UNIFORM)

# full graph with all the data for the test and validation purpose
full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(full_data.sources, full_data.destinations, full_data.edge_idxs, full_data.timestamps):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
full_ngh_finder = NeighborFinder(full_adj_list, uniform=UNIFORM)

train_rand_sampler = RandEdgeSampler(
    train_data.sources, train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations)
# nn_val_rand_sampler = RandEdgeSampler(nn_val_data.sources, nn_val_data.destinations)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations)
# nn_test_rand_sampler = RandEdgeSampler(nn_test_src_l, nn_test_data.destinations)


# Model initialize
tgan = TGAN(train_ngh_finder, node_features, edge_features,
            num_layers=NUM_LAYER, num_classes=NUM_CLASSES, use_time=USE_TIME, agg_method=AGG_METHOD,
            attn_mode=ATTN_MODE,
            seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT, node_dim=NODE_DIM, time_dim=TIME_DIM)

optimizer = torch.optim.Adam(tgan.parameters(), lr=LEARNING_RATE)
if args.loss == 'l1':
    criterion = torch.nn.L1Loss()
elif args.loss == 'cross_entropy':
    criterion = torch.nn.CrossEntropyLoss()
else:
    criterion = torch.nn.MSELoss()
tgan = tgan.to(device)

num_instance = len(train_data.sources)
num_batch = math.ceil(num_instance / BATCH_SIZE)

logger.info('num of training instances: {}'.format(num_instance))
logger.info('num of batches per epoch: {}'.format(num_batch))
idx_list = np.arange(num_instance)
np.random.shuffle(idx_list)
best_model_path = None
early_stopper = EarlyStopMonitor(higher_better=False)
for epoch in range(NUM_EPOCH):
    # Training
    # training use only training graph
    tgan.ngh_finder = train_ngh_finder
    mae_raw, mae, r2_raw, r2, m_loss = [], [], [], [], []
    np.random.shuffle(idx_list)
    logger.info('start {} epoch'.format(epoch))
    
    if(SCALE != 'Discr'):
        for k in tqdm(range(num_batch)):
            # percent = 100 * k / num_batch
            # if k % int(0.2 * num_batch) == 0:
            #     logger.info('progress: {0:10.4f}'.format(percent))

            s_idx = k * BATCH_SIZE
            e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
            src_l_cut, dst_l_cut = train_data.sources[s_idx:
                                                    e_idx], train_data.destinations[s_idx:e_idx]
            ts_l_cut = train_data.timestamps[s_idx:e_idx]
            label_l_cut = train_data.labels[s_idx:e_idx]
            raw_label_l_cut = train_data.raw_labels[s_idx:e_idx]
            size = len(src_l_cut)
            _, dst_l_fake = train_rand_sampler.sample(size)

            with torch.no_grad():
                pos_label = torch.tensor(label_l_cut, dtype=torch.float, device=device)
                neg_label = torch.zeros(size, dtype=torch.float, device=device)

            optimizer.zero_grad()
            tgan = tgan.train()
            pos_pred, neg_pred = tgan.contrast(
                src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)

            # if args.scale_label == 'none':
            loss = criterion(torch.concat(
                [pos_pred, neg_pred]), torch.concat([pos_label, neg_label]))
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
                pred_score = np.concatenate(
                    [(pos_pred).cpu().detach().numpy(), (neg_pred).cpu().detach().numpy()])
                true_label = np.concatenate([label_l_cut, np.zeros(size)])
                true_label_raw = np.concatenate([raw_label_l_cut, np.zeros(size)])

                mae.append(mean_absolute_error(true_label, pred_score))

                # f1.append(f1_score(true_label, pred_label))
                m_loss.append(loss.item())
                r2.append(r2_score(true_label, pred_score))
                if args.scale_label != 'none':
                    pred_score_raw = scaleUtil.convert_to_raw_label_scale(
                        np.concatenate([dst_l_cut, dst_l_fake]), pred_score)
                    mae_raw.append(mean_absolute_error(
                        true_label_raw, pred_score_raw))
                    r2_raw.append(r2_score(true_label_raw, pred_score_raw))

        # validation phase use all information
        tgan.ngh_finder = full_ngh_finder
        val_mae_raw, val_mae, val_r2_raw, val_r2 = eval_one_epoch(tgan, val_rand_sampler, val_data.sources,
                                                                val_data.destinations, val_data.timestamps,
                                                                val_data.labels, val_data.raw_labels)

    else:
        for k in tqdm(range(num_batch)):
    
            s_idx = k * BATCH_SIZE
            e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
            src_l_cut, dst_l_cut = train_data.sources[s_idx:
                                                    e_idx], train_data.destinations[s_idx:e_idx]
            
            ts_l_cut = train_data.timestamps[s_idx:e_idx]
            label_l_cut = train_data.labels[s_idx:e_idx]
            raw_label_l_cut = train_data.raw_labels[s_idx:e_idx]
            
        
            size = len(src_l_cut)
            
            _, dst_l_fake = train_rand_sampler.sample(size)

            with torch.no_grad():
                pos_label = torch.tensor(
                    label_l_cut, dtype=torch.float, device=device)
                neg_label = torch.zeros((size,NUM_CLASSES) , dtype=torch.float, device=device)
            
            optimizer.zero_grad()
            
            tgan = tgan.train()
            pos_pred, neg_pred = tgan.contrast(
                src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)
            
            loss = criterion(torch.concat(
                [pos_pred, neg_pred]), torch.concat([pos_label, neg_label]))
        
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                tgan = tgan.eval()
                pred_score_discrete = np.concatenate(
                    [(pos_pred).cpu().detach().numpy(), (neg_pred).cpu().detach().numpy()])
                pred_class = np.argmax(pred_score_discrete, axis=1)
                pred_score = scaleUtil.get_centroid_preds(np.concatenate([dst_l_cut, dst_l_fake]), pred_class)

                true_discrete_labels = np.concatenate([label_l_cut, np.zeros((size, num_classes))])
                true_labels_class = np.argmax(true_discrete_labels, axis=1)
                true_label = scaleUtil.get_centroid_preds(np.concatenate([dst_l_cut, dst_l_fake]), true_labels_class)
                
                true_label_raw = np.concatenate([raw_label_l_cut, np.zeros(size)])

                mae.append(mean_absolute_error(true_label.cpu().numpy(), pred_score.cpu().numpy()))

                m_loss.append(loss.item())
                r2.append(r2_score(true_label.cpu().numpy(), pred_score.cpu().numpy()))
                if SCALE != 'none':
                    mae_raw.append(mean_absolute_error(
                        true_label_raw, pred_score.cpu().numpy()))
                    r2_raw.append(r2_score(true_label_raw, pred_score.cpu().numpy()))
    
        tgan.ngh_finder = full_ngh_finder
        val_mae_raw, val_mae, val_r2_raw, val_r2 = eval_one_epoch(tgan, val_rand_sampler, val_data.sources,
                                                                val_data.destinations, val_data.timestamps,
                                                                val_data.labels, val_data.raw_labels)
                           

    logger.info('epoch: {}:'.format(epoch))
    logger.info('Epoch mean loss: {:.4f}'.format(np.mean(m_loss)))
    # logger.info('train acc: {}, val acc: {}, new node val acc: {}'.format(np.mean(acc), val_acc, nn_val_acc))
    # logger.info('train R2: {}, val R2: {}, new node val R2: {}'.format(np.mean(r2), val_auc, nn_val_auc))
    # logger.info('train MAE: {}, val ap: {}, new node val MAE: {}'.format(np.mean(mae), val_ap, nn_val_ap))
    # logger.info('train f1: {}, val f1: {}, new node val f1: {}'.format(np.mean(f1), val_f1, nn_val_f1))

    logger.info('train R2: {:.4f}, val R2: {:.4f}'.format(np.mean(r2), val_r2))
    logger.info('train MAE: {:.4f}, val MAE: {:.4f}'.format(
        np.mean(mae), val_mae))
    if args.scale_label != 'none':
        logger.info('train raw R2: {:.4f}, val raw R2: {:.4f}'.format(
            np.mean(r2_raw), val_r2_raw))
        logger.info('train raw MAE: {:.4f}, val raw MAE: {:.4f}'.format(
            np.mean(mae_raw), val_mae_raw))

    if early_stopper.early_stop_check(val_mae):
        logger.info('No improvment over {} epochs, stop training'.format(
            early_stopper.max_round))
        break
    else:
        if early_stopper.best_epoch == epoch:
            logger.info('Best epoch {} is the same as current epoch {}, save the model'.format(
                early_stopper.best_epoch, epoch))
            best_model_path = get_checkpoint_path(epoch)
            torch.save(tgan.state_dict(), best_model_path)
    store_checkpoint(epoch, tgan, optimizer)

# Load best model for testing
logger.info(
    f'Loading the best model at epoch {early_stopper.best_epoch} for Testing')
best_model_path = get_checkpoint_path(early_stopper.best_epoch)
tgan.load_state_dict(torch.load(best_model_path))
logger.info(f'Loaded the best model from {best_model_path} for Testing')
# testing phase use all information
tgan.ngh_finder = full_ngh_finder
test_mae_raw, test_mae, test_r2_raw, test_r2 = eval_one_epoch(tgan, test_rand_sampler, test_data.sources,
                                                              test_data.destinations, test_data.timestamps,
                                                              test_data.labels, test_data.raw_labels)

logger.info('Test R2: {:.4f}'.format(test_r2))
logger.info('Test MAE: {:.4f}'.format(test_mae))
if args.scale_label != 'none':
    logger.info('Test raw R2: {:.4f}'.format(test_r2_raw))
    logger.info('Test raw MAE: {:.4f}'.format(test_mae_raw))


def store_best_metrics(mae_raw, mae, r2_raw, r2):
    # check for the existence of the best metric file, if not, create a new file and store the best metrics
    if os.path.exists(BEST_METRICS_PATH):
        with open(BEST_METRICS_PATH, 'r') as f:
            lines = f.readlines()
            f.close()
        line_split = lines[1].split('\t')
        mae_raw_best = float(line_split[2])
        if mae_raw_best < mae_raw:
            logger.info(
                f'Existing raw MAE {mae_raw_best} in {BEST_METRICS_PATH} is better than new one {mae_raw}, not storing the new one')
            return
    with open(BEST_METRICS_PATH, 'w') as f:
        f.write('model_path\tmae\tmae_raw\tr2\tr2_raw\n')
        f.write('{}\t{}\t{}\t{}\t{}\n'.format(
            MODEL_SAVE_PATH, mae, mae_raw, r2, r2_raw))
        f.close()
    logger.info('Saving model')
    torch.save(tgan.state_dict(), MODEL_SAVE_PATH)
    logger.info('Model saved')


store_best_metrics(test_mae_raw, test_mae, test_r2_raw, test_r2)
