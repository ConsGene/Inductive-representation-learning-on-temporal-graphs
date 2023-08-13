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

class Data:
  def __init__(self, sources, destinations, dest_categories, timestamps, edge_idxs, raw_labels, labels):
    self.sources = sources
    self.destinations = destinations
    self.dest_categories = dest_categories
    self.timestamps = timestamps
    self.edge_idxs = edge_idxs
    self.raw_labels = raw_labels
    self.labels = labels



def get_data(dataset_name, scale_label, device):
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
        self.num_classes = num_classes

        if scale_label == 'MinMax':
            self.sscaler = preprocessing.MinMaxScaler
        elif scale_label == 'Quantile':
            self.scaler = preprocessing.QuantileTransformer
        elif scale_label == 'Log':
            self.scaler = preprocessing.StandardScaler
        elif scale_label == 'Cbrt':
            self.scaler = preprocessing.StandardScaler
        elif scale_label == 'Discr':
            self.scaler = KBinsDiscretizer

        self.device = device
        self.i2cat = None
        self.scalers_dict = None
    
    def transform_df(self, original_graph_df, valid_train_flag):
        graph_df = original_graph_df.copy()
        graph_df['raw_label'] = graph_df.label.copy()

        self.i2cat = graph_df.groupby('i').first(
        ).reset_index().set_index('i')['cat'].to_dict()

        for cat in self.i2cat.values():
            orig_labels = graph_df.loc[(
                                               graph_df.cat == cat) & valid_train_flag, 'label'].values
            lower = np.quantile(orig_labels, 0.001)
            upper = np.quantile(orig_labels, 0.999)
            graph_df.loc[(graph_df.cat == cat) & valid_train_flag, 'label'] = np.clip(orig_labels, lower, upper)
        
        if self.scale_label == 'none':
            label_l = graph_df.label.values
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
            train_df = graph_df[valid_train_flag]


        self.scalers_dict = {}
        if self.scale_label == 'Cbrt':
            graph_df.label = np.cbrt(graph_df.label.values)
            label_l = graph_df.label.values
            raw_label_l = graph_df.raw_label.values

        elif self.scale_label == 'Discr':

            columns = [f'label_{i}' for i in range(self.num_classes)]
            df = pd.DataFrame(
                np.zeros(shape=(graph_df.shape[0], self.num_classes)), columns=columns)
            graph_df = pd.concat([graph_df.reset_index(drop=True), df.reset_index(drop=True)], axis=1)

            uniq_cats = set(self.i2cat.values())
            for cat in uniq_cats:
                cat_train_df = train_df[train_df.cat == cat]
                self.scalers_dict[cat] = self.scaler(n_bins=self.num_classes)
                train_label_vals = self.prepare_transform(
                    cat_train_df.label.values)
                self.scalers_dict[cat].fit(train_label_vals.reshape(-1, 1))

                # Transform the label_vals for the current category and concatenate with graph_df.
                label_vals = self.prepare_transform(
                    graph_df.loc[graph_df.cat == cat]['label'].values)

                if label_vals.shape[0] == 0:
                    continue

                transformed_vals = self.scalers_dict[cat].transform(
                    label_vals.reshape(-1, 1))
                graph_df.loc[graph_df.cat == cat, columns] = transformed_vals
                label_l = graph_df[columns].to_numpy()
                raw_label_l = graph_df.raw_label.values

        else:
            for cat in self.i2cat.values():
                cat_train_df = train_df[train_df.cat == cat]
                self.scalers_dict[cat] = self.scaler()
                train_label_vals = self.prepare_transform(
                    cat_train_df.label.values)
                self.scalers_dict[cat].fit(train_label_vals.reshape(-1, 1))
                label_vals = self.prepare_transform(
                    graph_df.loc[graph_df.cat == cat]['label'].values)
                label_vals_list = label_vals.tolist()  # Convert numpy array to a list
                graph_df.loc[graph_df.cat == cat, 'label'] = self.scalers_dict[cat].transform(
                    np.array(label_vals_list).reshape(-1, 1))

                label_l = graph_df.label.values
                raw_label_l = graph_df.raw_label.values

        return label_l, raw_label_l
        
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
            cat = self.i2cat[dst]
            if self.scale_label == 'Cbrt':
                raw = np.power(pred, 3)
                raw_preds.append(raw)
            else:
                raw_pred = self.scalers_dict[cat].inverse_transform(
                    [pred.tolist()])  # Convert numpy array to a list
                if self.scale_label == 'Log':
                    raw = np.sign(raw) * (np.exp(np.abs(raw))-1)
                elif self.scale_label == 'Cbrt':
                    raw = np.power(raw, 3)
                raw_preds.append(raw[0, 0])
        if isinstance(preds, np.ndarray):
            return np.array(raw_preds)
        else:
            return torch.tensor(raw_preds, dtype=float, device=self.device)


class KBinsDiscretizer:
    def __init__(self, n_bins, strategy="kmeans"):
        self.n_bins = n_bins
        self.strategy = strategy

    def fit(self, X):
        X = X.flatten()
        # if strategy is kmeans, generte bin_edges_ with sklearn kmeans
        if self.strategy == "kmeans":
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.n_bins,
                            random_state=0).fit(X.reshape(-1, 1))
            self.bin_edges_ = np.sort(kmeans.cluster_centers_.flatten())
            # save the standard deviation of each cluster into self.std
            self.std_ = np.zeros(self.n_bins)
            for i in range(self.n_bins):
                self.std_[i] = np.std(X[kmeans.labels_ == i])
        else:
            self.bin_edges_ = np.percentile(
                X, np.linspace(0, 100, self.n_bins - 1))
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
            X_binned = np.digitize(X, self.bin_edges_, right=True) - 1
        # Create one-hot encoding
        one_hot = np.zeros((len(X), self.n_bins))
        one_hot[np.arange(len(X)), X_binned] = 1
        return one_hot

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return self.bin_edges_[X - 1]

    def __repr__(self):
        return "KBinsDiscretizer(n_bins={})".format(self.n_bins)
