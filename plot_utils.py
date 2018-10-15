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
	optimizer = ['sgd' if len(n.split("_")) == 3 else "adam" for n in modelname]
	df_neumf['dl_frame'] = dl_frame
	df_neumf['with_pretrained'] = with_pretrained
	df_neumf['last_layer'] = last_layer
	df_neumf['optimizer'] = optimizer
	df_neumf.drop(['modelname', 'train_time'], axis=1, inplace=True)
	df_neumf.sort_values('dl_frame').reset_index(drop=True, inplace=True)
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
	df_time_gmf = df[(df.modelname.str.contains('GMF')) &
		(df.modelname.str.contains('bs_256')) &
		(df.modelname.str.contains('lr_001')) ]
	modelname = df_time_gmf.modelname.tolist()
	modelname = [re.sub(".h5|.pt|.params|_reg_00|reg_00", "", n) for n in modelname]
	dl_frame = [n.split("_")[0] for n in modelname]
	n_emb = [int(n.split("_")[-1]) for n in modelname]
	df_time_gmf['dl_frame'] = dl_frame
	df_time_gmf['n_emb'] = n_emb
	df_time_gmf.drop(['modelname','best_hr','best_ndcg','best_iter'], axis=1, inplace=True)
	df_time_gmf = df_time_gmf.sort_values('dl_frame')
	df_time_gmf.reset_index(drop=True, inplace=True)

	df_time_mlp = df[(df.modelname.str.contains('MLP')) &
		(df.modelname.str.contains('bs_256')) &
		(df.modelname.str.contains('wodp')) ]
	modelname = df_time_mlp.modelname.tolist()
	modelname = [re.sub(".h5|.pt|.params|_reg_00|reg_00", "", n) for n in modelname]
	dl_frame = [n.split("_")[0] for n in modelname]
	n_emb = [int(n.split("_")[8]) for n in modelname]
	df_time_mlp['dl_frame'] = dl_frame
	df_time_mlp['n_emb'] = n_emb
	df_time_mlp.drop(['modelname','best_hr','best_ndcg','best_iter'], axis=1, inplace=True)
	df_time_mlp = df_time_mlp.drop_duplicates(['dl_frame','n_emb'], keep='last')
	df_time_mlp = df_time_mlp.sort_values(['dl_frame', 'n_emb'])
	df_time_mlp.reset_index(drop=True, inplace=True)

	return df_time_gmf, df_time_mlp

def plot_train_time(df):
	df_time_gmf, df_time_mlp = TIME_df(df)
	plt.figure(figsize=(15, 12))
	plt.subplot(2,1,1)
	plt.subplots_adjust(hspace=0.3)
	fig = sns.barplot(x='n_emb', y='train_time', hue='dl_frame', data=df_time_gmf)
	sns.set_context("notebook", font_scale=1.5)
	fig.set(ylabel="Training Time")
	fig.set(xlabel="Number of Embeddings")
	fig.set(title="GMF")
	plt.legend(loc="upper center")

	plt.subplot(2,1,2)
	fig = sns.barplot(x='n_emb', y='train_time', hue='dl_frame', data=df_time_mlp)
	sns.set_context("notebook", font_scale=1.5)
	fig.set(ylabel="Training Time")
	fig.set(xlabel="Number of Embeddings")
	fig.set(title="MLP")
	plt.legend(loc="upper center")
