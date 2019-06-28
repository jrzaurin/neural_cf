"""
This is a series of simple helpers that are coded here so that the  notebook
with the summary of the results is cleaner and more readable.

There is nothing particularly relevant here.

@author: Javier Rodriguez (jrzaurin@gmail.com)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re


def GMF_df(df):
	df_gmf = df[df.modelname.str.contains('GMF')]
	modelname = df_gmf.modelname
	modelname = [re.sub(".h5|.pt|.params|_reg_00", "", n) for n in modelname]
	dl_frame = [n.split("_")[0] for n in modelname]
	n_emb = [int(n.split("_")[-1]) for n in modelname]
	df_gmf['dl_frame'] = dl_frame
	df_gmf['n_emb'] = n_emb
	df_gmf.drop(['modelname', 'train_time'], axis=1, inplace=True)
	df_gmf.reset_index(drop=True, inplace=True)
	idx = df_gmf.groupby(['dl_frame','n_emb'])['best_hr'].transform(max) == df_gmf['best_hr']
	df_gmf = df_gmf[idx].sort_values(['dl_frame','n_emb']).reset_index(drop=True)
	return df_gmf


def MLP_df(df):
	df_mlp = df[df.modelname.str.contains('MLP')]
	modelname = df_mlp.modelname
	modelname = [re.sub(".h5|.pt|.params|_reg_00|reg_00", "", n) for n in modelname]
	dl_frame = [n.split("_")[0] for n in modelname]
	n_emb = [int(n.split("_")[8]) for n in modelname]
	with_dropout = [n.split("_")[-1] for n in modelname]
	df_mlp['dl_frame'] = dl_frame
	df_mlp['n_emb'] = n_emb
	df_mlp['with_dropout'] = with_dropout
	df_mlp.drop(['modelname', 'train_time'], axis=1, inplace=True)
	df_mlp.reset_index(drop=True, inplace=True)
	idx = df_mlp.groupby(['dl_frame','n_emb'])['best_hr'].transform(max) == df_mlp['best_hr']
	df_mlp = df_mlp[idx].sort_values(['dl_frame','n_emb']).reset_index(drop=True)
	return df_mlp


def NeuMF_df(df):
	df_neumf = df[df.modelname.str.contains('NeuMF')]
	modelname = df_neumf.modelname
	modelname = [re.sub(".h5|.pt|.params|_reg_00|reg_00|_NeuMF", "", n) for n in modelname]
	dl_frame = [n.split("_")[0] for n in modelname]
	with_pretrained = [n.split("_")[1] for n in modelname]
	last_layer = [n.split("_")[2] for n in modelname]
	df_neumf['dl_frame'] = dl_frame
	df_neumf['with_pretrained'] = with_pretrained
	df_neumf['last_layer'] = last_layer
	df_neumf.drop(['modelname', 'train_time'], axis=1, inplace=True)
	df_neumf = df_neumf.sort_values('dl_frame')
	df_neumf.reset_index(drop=True, inplace=True)
	return df_neumf


def plot_metrics(df_gmf, df_mlp):
	plt.figure(figsize=(14, 9))
	plt.subplot(2,2,1)
	plt.subplots_adjust(hspace=0.4)
	fig = sns.pointplot(x='n_emb', y='best_hr', hue='dl_frame',data=df_gmf,
	                    linestyles=['-', '--', '-.'],
	                    markers = ["o", "s", "^"])
	sns.set_context("notebook", font_scale=1.5)
	fig.set(ylabel="HR@10")
	fig.set(xlabel="Number of Embeddings")
	fig.set(title="GMF")
	plt.legend(loc="upper left")
	plt.subplot(2,2,2)
	fig = sns.pointplot(x='n_emb', y='best_ndcg', hue='dl_frame',data=df_gmf,
	                    linestyles=['-', '--', '-.'],
	                    markers = ["o", "s", "^"])
	sns.set_context("notebook", font_scale=1.5)
	fig.set(ylabel="NDCG@10")
	fig.set(xlabel="Number of Embeddings")
	fig.set(title="GMF")
	plt.legend(loc="upper left")
	plt.subplot(2,2,3)
	fig = sns.pointplot(x='n_emb', y='best_hr', hue='dl_frame',data=df_mlp,
	                    linestyles=['-', '--', '-.'],
	                    markers = ["o", "s", "^"])
	sns.set_context("notebook", font_scale=1.5)
	fig.set(ylabel="HR@10")
	fig.set(xlabel="Number of Embeddings")
	fig.set(title="MLP")
	plt.legend(loc="upper left")
	plt.subplot(2,2,4)
	fig = sns.pointplot(x='n_emb', y='best_ndcg', hue='dl_frame',data=df_mlp,
	                    linestyles=['-', '--', '-.'],
	                    markers = ["o", "s", "^"])
	sns.set_context("notebook", font_scale=1.5)
	fig.set(ylabel="NDCG@10")
	fig.set(xlabel="Number of Embeddings")
	fig.set(title="MLP")
	plt.legend(loc="upper left")


def TIME_df(df):
	tmp_df = df[df.modelname.str.contains('GMF|MLP')]
	modelname = tmp_df.modelname
	modelname = [re.sub(".h5|.pt|.params|_reg_00", "", n) for n in modelname]
	dl_frame = [n.split("_")[0] for n in modelname]
	model = [n.split("_")[1] for n in modelname]
	n_emb_locs = [np.where([s =='emb' for s in m.split("_")])[0][0]+1 for m in modelname]
	n_emb = [int(n.split("_")[i]) for n,i in zip(modelname, n_emb_locs)]
	bs_locs = [np.where([s =='bs' for s in m.split("_")])[0][0]+1 for m in modelname]
	bs = [int(n.split("_")[i]) for n,i in zip(modelname, bs_locs)]
	tmp_df['dl_frame'] = dl_frame
	tmp_df['model'] = model
	tmp_df['n_emb'] = n_emb
	tmp_df['bs'] = bs

	df_gmf = tmp_df[tmp_df.model.str.contains('GMF')]
	df_gmf = df_gmf[df_gmf.n_emb==8]
	df_gmf.drop(['modelname', 'best_hr', 'best_ndcg', 'best_iter'], axis=1, inplace=True)
	df_gmf.reset_index(drop=True, inplace=True)
	idx = df_gmf.groupby(['dl_frame','bs'])['train_time'].transform(min) == df_gmf['train_time']
	df_gmf = df_gmf[idx].sort_values(['dl_frame','bs']).reset_index(drop=True)

	df_mlp = tmp_df[(tmp_df.model.str.contains('MLP')) & (tmp_df.bs==256)]
	df_mlp.drop(['modelname', 'best_hr', 'best_ndcg', 'best_iter'], axis=1, inplace=True)
	df_mlp.reset_index(drop=True, inplace=True)
	idx = df_mlp.groupby(['dl_frame','bs', 'n_emb'])['train_time'].transform(min) == df_mlp['train_time']
	df_mlp = df_mlp[idx].sort_values(['dl_frame','bs']).reset_index(drop=True)
	df_mlp = df_mlp[df_mlp.dl_frame != 'gluon'].reset_index()

	return df_gmf, df_mlp

def plot_train_time(df):
	df_time_gmf, df_time_mlp = TIME_df(df)
	plt.figure(figsize=(15, 12))
	plt.subplot(2,1,1)
	plt.subplots_adjust(hspace=0.3)
	fig = sns.barplot(x='bs', y='train_time', hue='dl_frame', data=df_time_gmf)
	sns.set_context("notebook", font_scale=1.5)
	fig.set(ylabel="Training Time")
	fig.set(xlabel="Batch Size")
	fig.set(title="GMF")
	plt.legend(loc="upper center")

	plt.subplot(2,1,2)
	fig = sns.barplot(x='n_emb', y='train_time', hue='dl_frame', data=df_time_mlp)
	sns.set_context("notebook", font_scale=1.5)
	fig.set(ylabel="Training Time")
	fig.set(xlabel="Number of Embeddings")
	fig.set(title="MLP")
	plt.legend(loc="upper center")
