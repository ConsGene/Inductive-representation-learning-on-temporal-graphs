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
from sklearn.cluster import AgglomerativeClustering, KMeans

class Data:
  def __init__(self, sources, destinations, dest_categories, timestamps, edge_idxs, raw_labels, labels):
    self.sources = sources
    self.destinations = destinations
    self.dest_categories = dest_categories
    self.timestamps = timestamps
    self.edge_idxs = edge_idxs
    self.raw_labels = raw_labels
    self.labels = labels

class KBinsDiscretizer:
    def __init__(self, n_bins, strategy="agg"):
        self.n_bins = n_bins
        self.strategy = strategy

    def fit(self, X):
        X = X.flatten()
        # if strategy is kmeans, generte bin_edges_ with sklearn kmeans
        if self.strategy == "kmeans" or self.strategy == "agg":
            # reduce the n_cluster to ensure the first bin is always 0
            n_clusters = self.n_bins - 1
            if self.strategy == "kmeans":
                clustering_model = KMeans(n_clusters=n_clusters, random_state=0)
            else:
                clustering_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
            clustering_model.fit(X.reshape(-1, 1))
            cluster_labels, bin_edges_ = zip(*sorted(zip(clustering_model.labels_, clustering_model.cluster_centers_.flatten()), key=lambda x: x[1]))
            # ensure the first bin is always 0
            self.bin_edges_ = np.concatenate(([0], bin_edges_))
            # save the standard deviation of each cluster into self.std
            self.std_ = np.zeros(self.n_bins)
            for i in range(n_clusters):
                self.std_[i+1] = np.std(X[cluster_labels == i])
            
        else:    
            self.bin_edges_ = np.percentile(X, np.linspace(0, 100, self.n_bins - 1))
        return self

    def transform(self, X):
        X = X.flatten()
        # Assign each value to a bin
        # if strategy is kmeans, assign X_binned to the closest bin_edges_ 
        if self.strategy == "kmeans":
            X_binned = np.zeros(len(X), dtype=int)
            for i in range(len(X)):
                X_binned[i] = np.argmin(np.abs(X[i] - self.bin_edges_))
        else:
            X_binned = np.digitize(X, self.bin_edges_, right=True)
        # Create one-hot encoding
        one_hot = np.zeros((len(X), self.n_bins))
        one_hot[np.arange(len(X)), X_binned] = 1
        return one_hot

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return self.bin_edges_[np.argmax(X)]

    def __repr__(self):
        return "KBinsDiscretizer(n_bins={})".format(self.n_bins)

def get_data(dataset_name, scale_label, device, num_classes=10):
    ### Load data and train val test split
    graph_df = pd.read_csv('./processed/ml_{}.csv'.format(dataset_name))
    edge_features = np.load('./processed/ml_{}.npy'.format(dataset_name))
    node_features = np.load('./processed/ml_{}_node.npy'.format(dataset_name)) 

    val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

    ts_l = graph_df.ts.values
    valid_train_flag = (ts_l <= val_time)


    train_df = graph_df[valid_train_flag]
    train_i = train_df.i.unique()
    # filter out new merchants in the valid/test
    from scipy import stats

    graph_df = graph_df[graph_df.i.isin(train_i)]

    ts_l = graph_df.ts.values
    valid_train_flag = (ts_l <= val_time)
    src_l = graph_df.u.values
    dst_l = graph_df.i.values
    cat_l = graph_df.cat.values
    e_idx_l = graph_df.idx.values
    # scaling labels
    scaleUtil = ScaleUtil(scale_label, device, num_classes=num_classes)
    label_l, raw_label_l = scaleUtil.transform_df(graph_df, valid_train_flag)

    full_data = Data(sources=src_l, 
            destinations=dst_l, 
            dest_categories=cat_l, 
            timestamps=ts_l, 
            edge_idxs=e_idx_l, 
            raw_labels=raw_label_l, 
            labels=label_l)

    def get_dataset(flag):
        ds = Data(sources=src_l[flag], 
            destinations=dst_l[flag], 
            dest_categories=cat_l[flag], 
            timestamps=ts_l[flag], 
            edge_idxs=e_idx_l[flag], 
            raw_labels=raw_label_l[flag], 
            labels=label_l[flag])
        return ds
    
    train_data = get_dataset(valid_train_flag)
    # select validation and test dataset
    valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
    val_data = get_dataset(valid_val_flag)

    valid_test_flag = ts_l > test_time
    test_data = get_dataset(valid_test_flag)
    return node_features, edge_features, full_data, train_data, val_data, test_data, scaleUtil


class ScaleUtil:
    def __init__(self, scale_label, device, num_classes=10):
        self.scale_label = scale_label
        if scale_label == 'MinMax':
            self.sscaler = preprocessing.MinMaxScaler
        elif scale_label == 'Quantile':
            self.scaler = preprocessing.QuantileTransformer
        elif scale_label == 'Log':
            self.scaler = preprocessing.StandardScaler
        elif scale_label == 'Cbrt':
            self.scaler = preprocessing.StandardScaler
        elif scale_label.startswith('Discr'):
            self.scaler = KBinsDiscretizer
            self.num_classes = num_classes

        self.device = device
        self.i2cat = None
        self.scalers_dict = None
    
    def transform_df(self, original_graph_df, valid_train_flag):
        graph_df = original_graph_df.copy()
        graph_df['raw_label'] = graph_df.label.copy()
        
        self.i2cat = graph_df.groupby('i').first().reset_index().set_index('i')['cat'].to_dict()
        # TODO: whether to clip the label?
        for cat in set(self.i2cat.values()):
            orig_labels = graph_df.loc[(graph_df.cat == cat) & valid_train_flag, 'label'].values
            lower = np.quantile(orig_labels, 0.001)
            upper = np.quantile(orig_labels, 0.999)
            graph_df.loc[(graph_df.cat == cat) & valid_train_flag, 'label'] = np.clip(orig_labels, lower, upper)
        
        if self.scale_label == 'none':
            # no need to scale
            return graph_df.label.values, graph_df.label.values
        scaled_label_cols = 'label'
        if self.scale_label == 'Cbrt':
            graph_df.label = np.cbrt(graph_df.label.values)
        else:
            graph_df['label'] = self.prepare_transform(graph_df.label.values)
            if self.scale_label.startswith('Discr'):
                 scaled_label_cols=[f'label_{i}' for i in range(self.num_classes)]
                 graph_df[scaled_label_cols] = np.zeros(shape=(graph_df.shape[0],self.num_classes))
            train_df = graph_df[valid_train_flag]
            self.scalers_dict = {}
            if self.scale_label.endswith('#all'):
                train_label_vals = train_df.label.values
                self.scalers_dict['#all'] = self.scaler(n_bins=self.num_classes, strategy=self.scale_label.split('-')[1])
                self.scalers_dict['#all'].fit(train_label_vals.reshape(-1, 1))
                label_vals = graph_df.label.values      
                graph_df[scaled_label_cols] = self.scalers_dict['#all'].transform(label_vals.reshape(-1, 1))
            else:
                for cat in set(self.i2cat.values()):
                    if self.scale_label.startswith('Discr'):
                        self.scalers_dict[cat] = self.scaler(n_bins=self.num_classes, strategy=self.scale_label.split('-')[1])
                    else:
                        self.scalers_dict[cat] = self.scaler()
                    train_label_vals = train_df[train_df.cat==cat].label.values
                    self.scalers_dict[cat].fit(train_label_vals.reshape(-1, 1))
                    label_vals = graph_df.loc[graph_df.cat == cat].label.values      
                    graph_df.loc[graph_df.cat == cat, scaled_label_cols] = self.scalers_dict[cat].transform(label_vals.reshape(-1, 1))
        return graph_df[scaled_label_cols].values, graph_df.raw_label.values
        
    def prepare_transform(self, label_vals):
        if self.scale_label == 'Log':
            label_vals = np.sign(label_vals) * np.log(np.abs(label_vals)+1)
        elif self.scale_label == 'Cbrt':
            label_vals = np.cbrt(label_vals)
        return label_vals

    def convert_to_raw_label_scale(self, dst_l_cut, preds):
        if (self.i2cat is None) or (self.scalers_dict is None):
            raise RuntimeError("self.i2cat or self.scalers_dict is None. Run ScaleUtil.transform_df function first")
        raw_preds = []
        for dst, pred in zip(dst_l_cut, preds):
            if self.scale_label == 'Cbrt':
                raw = np.power(pred, 3)
                raw_preds.append(raw)
            else:
                if self.scale_label.endswith('#all'):
                    cat = '#all'
                else:
                    cat = self.i2cat[dst]
                raw = self.scalers_dict[cat].inverse_transform(pred.reshape(1, -1))
                if self.scale_label == 'Log':
                    raw = np.sign(raw) * (np.exp(np.abs(raw))-1)
                elif self.scale_label == 'Cbrt':
                    raw = np.power(raw, 3)
                if len(raw.shape) > 1:
                    raw = raw.item()
                raw_preds.append(raw)
        if isinstance(preds, np.ndarray):
            return np.array(raw_preds)
        else:
            return torch.tensor(raw_preds, dtype=float, device=self.device)