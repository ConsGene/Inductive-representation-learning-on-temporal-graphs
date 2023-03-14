import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# def preprocess(data_name):
#   u_list, i_list, ts_list, label_list = [], [], [], []
#   feat_l = []
#   idx_list = []

#   raw_df = pd.read_csv(data_name)
#   df = raw_df.groupby(['member_id','merchant', 'optimized_date']).agg({'transaction_amount':'sum', 'member_home_state':'first', 'category':'first', 'subcategory':'first', 'merchant_format_name':'first'}).reset_index()
#   df['ts'] = df['optimized_date'].apply(pd.to_datetime).astype(int) / 10**9
#   df = df.sort_values('ts')
#   result_df = pd.DataFrame({'u': df['member_id'].astype('category').cat.codes.tolist(),
#                             'i': df['merchant'].astype('category').cat.codes.tolist(),
#                             'ts': df['ts'].tolist(),
#                             'label': df['transaction_amount'].tolist(),
#                             'idx': df.index.values.tolist()})
#   return result_df, np.array([df['category'].astype('category').cat.codes.tolist(), df['member_home_state'].astype('category').cat.codes.tolist(), df['subcategory'].astype('category').cat.codes.tolist(), df['merchant_format_name'].astype('category').cat.codes.tolist()]).T
WEEK_IN_SECS = 7*24*60*60

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('whaleloops/phrase-bert')

enc_dict = {}
def encode(sentences):
    phrase_embs = model.encode(sentences)
    return phrase_embs
# def encode(sentences):

def transform_and_normalize(vecs, kernel, bias):
    """
        Applying transformation then standardize
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return normalize(vecs)
    
def normalize(vecs):
    """
        Standardization
    """
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5
    
def compute_kernel_bias(vecs):
    """
    Calculate Kernal & Bias for the final transformation - y = (x + bias).dot(kernel)
    """
    vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(s**0.5))
    W = np.linalg.inv(W.T)
    return W, -mu

def embed(df_col, dim):
    '''
        This method will accept array of sentences, roberta tokenizer & model
        next it will call methods for dimention reduction
    '''

    uniq_text = df_col.unique()
    vecs = encode(uniq_text)
    #Finding Kernal
    kernel, bias = compute_kernel_bias([vecs])
    kernel = kernel[:, :dim]
    #If you want to reduce it to 128 dim
    #kernel = kernel[:, :128]
    embeddings = []
    embeddings = np.vstack(vecs)

    #Sentence embeddings can be converted into an identity matrix
    #by utilizing the transformation matrix
    embeddings = transform_and_normalize(embeddings, 
                kernel=kernel,
                bias=bias
            )
    text2emb = {k:v for k, v in zip(uniq_text, embeddings)}
    all_emb = [text2emb[text] for text in df_col.tolist()]
    return np.array(all_emb)


def preprocess(data_name):
  u_list, i_list, ts_list, label_list = [], [], [], []
  feat_l = []
  idx_list = []

  raw_df = pd.read_csv(data_name)
  
  raw_df['ts'] = raw_df['optimized_date'].apply(pd.to_datetime).astype(int) // 10**9 // (1*WEEK_IN_SECS)
  df = raw_df.groupby(['member_id','merchant_format_name', 'ts']).agg({'transaction_amount':'sum', 'category':'first', 'subcategory':'first', 'member_home_state':'first'}).reset_index()
  df = df.sort_values('ts')
  result_df = pd.DataFrame({'u': df['member_id'].astype('category').cat.codes.tolist(),
                            'i': df['merchant_format_name'].astype('category').cat.codes.tolist(),
                            'name': df['merchant_format_name'].tolist(),
                            'ts': df['ts'].tolist(),
                            'label': df['transaction_amount'].tolist(),
                            'cat': df['category'].astype('category').tolist(),
                            'subcat': df['subcategory'].astype('category').tolist(),
                            'state': df['member_home_state'].astype('category').tolist(),
                            'idx': df.index.values.tolist()})
  # return result_df, np.array([df['category'].astype('category').cat.codes.tolist(), df['member_home_state'].astype('category').cat.codes.tolist()]).T
  return result_df #, pd.get_dummies(df['member_home_state'], prefix='_state'), pd.get_dummies(df['category'], prefix='_cat')

def reindex(df, bipartite=True):
  new_df = df.copy()
  if bipartite:
    assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

    upper_u = df.u.max() + 1
    # item index starts from (max user index + 1)
    new_i = df.i + upper_u

    new_df.i = new_i
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
  else:
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

  return new_df


def run(data_name, agg, bipartite=True):
  Path("data/").mkdir(parents=True, exist_ok=True)
  PATH = '../data/{}.csv'.format(data_name)
  OUT_DF = './processed/ml_{}_{}.csv'.format(data_name, agg)
  OUT_FEAT = './processed/ml_{}_{}.npy'.format(data_name, agg)
  OUT_NODE_FEAT = './processed/ml_{}_{}_node.npy'.format(data_name, agg)

  df = preprocess(PATH)
  df['cat_subcat'] = df[['cat','subcat']].agg(' - '.join, axis=1)
  df.fillna(' ', inplace=True)
  new_df = reindex(df, bipartite)

  u_group_df = new_df.groupby('u').first().reset_index().set_index('u')
  u_feat = embed(u_group_df['state'], 64)

  i_group_df = new_df.groupby('i').first().reset_index().set_index('i')
  cat = embed(i_group_df['cat_subcat'], 16)
  name = embed(i_group_df['name'], 48)
  i_feat = np.concatenate([cat, name], axis=1)

  max_idx = max(new_df.u.max(), new_df.i.max())
  node_feat_dim = max(u_feat.shape[1], i_feat.shape[1])
  if node_feat_dim % 2 != 0:
    node_feat_dim += 1
  node_feat = np.zeros((max_idx + 1, node_feat_dim))
  node_feat[1:new_df.u.max()+1, :u_feat.shape[1]] = u_feat
  node_feat[new_df.u.max()+1:, :i_feat.shape[1]] = i_feat
  new_df.to_csv(OUT_DF)

  feat = np.zeros((len(new_df), node_feat_dim))
  empty = np.zeros(feat.shape[1])[np.newaxis, :]
  feat = np.vstack([empty, feat])
  np.save(OUT_FEAT, feat)
  np.save(OUT_NODE_FEAT, node_feat)

parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='u2k_i200')
parser.add_argument('--agg', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='1W')
parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')

args = parser.parse_args()

run(args.data, args.agg)
