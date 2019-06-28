"""
@author: Javier Rodriguez (jrzaurin@gmail.com)
"""

import numpy as np
import pandas as pd
import os
import heapq
import argparse
import mxnet as mx

from mxnet import gluon, autograd, ndarray
from mxnet.gluon import Block, nn, HybridBlock
from GMF_gluon import GMF, train, evaluate, checkpoint
from MLP_gluon import MLP

from Dataset import Dataset as ml1mDataset
from time import time
from utils import *

import pdb

def parse_args():
    parser = argparse.ArgumentParser()

    # dirnames
    parser.add_argument("--datadir", type=str, default="Data_Javier",
        help="data directory.")
    parser.add_argument("--modeldir", type=str, default="models",
        help="models directory")
    parser.add_argument("--dataname", type=str, default="ml-1m",
        help="chose a dataset.")

    # general parameter
    parser.add_argument("--epochs", type=int, default=20,
        help="number of epochs.")
    parser.add_argument("--batch_size", type=int, default=256,
        help="batch size.")
    parser.add_argument("--lr", type=float, default=0.001,
        help="learning rate.")
    parser.add_argument("--learner", type=str, default="adam",
        help="Specify an optimizer: adagrad, adam, rmsprop, sgd")

    # GMF set up
    parser.add_argument("--n_emb", type=int, default=8,
        help="embedding size for the GMF part.")

    # MLP set up
    parser.add_argument("--layers", type=str, default="[64,32,16,8]",
        help="layer architecture. The first elements is used for the embedding \
        layers for the MLP part and equals n_emb*2")
    parser.add_argument("--dropouts", type=str, default="[0,0,0]",
        help="dropout per dense layer. len(dropouts) = len(layers)-1")

    # regularization
    parser.add_argument("--l2reg", type=float, default=0.,
        help="l2 regularization.")

    # Pretrained model names
    parser.add_argument("--freeze", type=int, default=0,
        help="freeze all but the last output layer where \
        weights are combined")
    parser.add_argument("--mf_pretrain", type=str, default="",
        help="Specify the pretrain model filename for GMF part. \
        If empty, no pretrain will be used")
    parser.add_argument("--mlp_pretrain", type=str, default="",
        help="Specify the pretrain model filename for MLP part. \
        If empty, no pretrain will be used")

    # Experiment set up
    parser.add_argument("--validate_every", type=int, default=1,
        help="validate every n epochs")
    parser.add_argument("--save_model", type=int , default=1)
    parser.add_argument("--n_neg", type=int, default=4,
        help="number of negative instances to consider per positive instance.")
    parser.add_argument("--topK", type=int, default=10,
        help="number of items to retrieve for recommendation.")

    return parser.parse_args()


class NeuMF(HybridBlock):
    def __init__(self, n_user, n_item, n_emb, layers, dropouts):
        super(NeuMF, self).__init__()

        self.layers = layers
        self.n_layers = len(layers)
        self.dropouts = dropouts
        self.n_user = n_user
        self.n_item = n_item

        self.mlp = gluon.nn.HybridSequential()
        with self.name_scope():
            self.mf_embeddings_user = nn.Embedding(n_user, n_emb, weight_initializer='normal', prefix="mf_embeddings_user_")
            self.mf_embeddings_item = nn.Embedding(n_item, n_emb, weight_initializer='normal', prefix="mf_embeddings_item_")
            self.mlp_embeddings_user = nn.Embedding(n_user, int(layers[0]/2), weight_initializer='normal', prefix = "mlp_embeddings_user_")
            self.mlp_embeddings_item = nn.Embedding(n_item, int(layers[0]/2), weight_initializer='normal', prefix = "mlp_embeddings_item_")
            for i in range(1,self.n_layers):
                self.mlp.add(nn.Dense(in_units=layers[i-1], units=layers[i], activation="relu", prefix="linear{}".format(i)))
                self.mlp.add(nn.Dropout(rate=dropouts[i-1]))

        with self.name_scope():
            self.out = nn.Dense(in_units=n_emb+layers[-1], units=1, weight_initializer='uniform', prefix="out_")

    def forward(self, users, items):

        mf_user_emb = self.mf_embeddings_user(users)
        mf_item_emb = self.mf_embeddings_item(items)

        mlp_user_emb = self.mlp_embeddings_user(users)
        mlp_item_emb = self.mlp_embeddings_item(items)

        mf_emb_vector = mf_user_emb*mf_item_emb
        mlp_emb_vector = ndarray.concat(mlp_user_emb,mlp_item_emb,dim=1)
        mlp_emb_vector = self.mlp(mlp_emb_vector)

        emb_vector = ndarray.concat(mf_emb_vector,mlp_emb_vector, dim=1)
        preds = self.out(emb_vector)

        return preds


def load_pretrain_model(model, gmf_model, mlp_model, layers):

    # MF embeddings
    model.mf_embeddings_item.weight.set_data(gmf_model.embeddings_item.weight.data())
    model.mf_embeddings_user.weight.set_data(gmf_model.embeddings_user.weight.data())

    # MLP embeddings
    model.mlp_embeddings_item.weight.set_data(mlp_model.embeddings_item.weight.data())
    model.mlp_embeddings_user.weight.set_data(mlp_model.embeddings_user.weight.data())

    # MLP layers
    model_dict =  model.collect_params()
    mlp_layers_dict = mlp_model.collect_params()
    mlp_layers_dict = {k: v for k, v in mlp_layers_dict.items() if 'linear' in k}
    for idx in range(1, len(layers)):
        newmf_layer_w = model.name + '_linear' + str(idx) + "weight"
        newmf_layer_b = model.name + '_linear' + str(idx) + "bias"
        mlp_layer_w = mlp_model.name + '_linear' + str(idx) + "weight"
        mlp_layer_b = mlp_model.name + '_linear' + str(idx) + "bias"
        model_dict[newmf_layer_w].set_data(mlp_layers_dict[mlp_layer_w].data())
        model_dict[newmf_layer_b].set_data(mlp_layers_dict[mlp_layer_b].data())

    # Prediction weights
    mf_prediction_weight, mf_prediction_bias = gmf_model.out.weight.data(), gmf_model.out.bias.data()
    mlp_prediction_weight, mlp_prediction_bias = mlp_model.out.weight.data(), mlp_model.out.bias.data()

    new_weight = ndarray.concat(mf_prediction_weight, mlp_prediction_weight, dim=1)
    new_bias = mf_prediction_bias + mlp_prediction_bias
    model.out.weight.set_data(0.5*new_weight)
    model.out.bias.set_data(0.5*new_bias)

    return model


if __name__ == '__main__':
    args = parse_args()

    datadir = args.datadir
    dataname = args.dataname
    modeldir = args.modeldir

    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    learner = args.learner

    n_emb = args.n_emb

    layers = eval(args.layers)
    dropouts = eval(args.dropouts)

    freeze = bool(args.freeze)
    mf_pretrain = os.path.join(modeldir, args.mf_pretrain)
    mlp_pretrain = os.path.join(modeldir, args.mlp_pretrain)
    with_pretrained = "wpret" if os.path.isfile(mf_pretrain) else "wopret"
    is_frozen = "frozen" if freeze else "trainable"

    l2reg = args.l2reg

    validate_every = args.validate_every
    save_model = bool(args.save_model)
    n_neg = args.n_neg
    topK = args.topK

    modelfname = "gluon_NeuMF" + \
        "_" + with_pretrained + \
        "_" + is_frozen + \
        "_" + learner + \
        ".params"
    modelpath = os.path.join(modeldir, modelfname)
    resultsdfpath = os.path.join(modeldir, 'results_df.p')

    dataset = ml1mDataset(os.path.join(datadir, dataname))
    trainRatings, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    n_users, n_items = trainRatings.shape

    test = get_test_instances(testRatings, testNegatives)
    test_dataset = mx.gluon.data.dataset.ArrayDataset(test)
    test_loader = mx.gluon.data.DataLoader(dataset=test_dataset,
        batch_size=100,
        shuffle=False)

    ctx =  mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
    model = NeuMF(n_users, n_items, n_emb, layers, dropouts)
    model.hybridize()
    model.initialize(ctx=ctx)
    # pdb.set_trace()
    if os.path.isfile(mf_pretrain) and os.path.isfile(mlp_pretrain):
        gmf_model = GMF(n_users, n_items, n_emb)
        gmf_model.load_parameters(mf_pretrain, ctx=ctx)
        mlp_model = MLP(n_users, n_items, layers, dropouts)
        mlp_model.load_parameters(mlp_pretrain, ctx=ctx)
        model = load_pretrain_model(model, gmf_model, mlp_model, layers)
        print("Load pretrained GMF {} and MLP {} models done. ".format(mf_pretrain, mlp_pretrain))

    if freeze:
        for param in model.collect_params().values():
            if not ('out' in param.name):
                param.grad_req = 'null'

    # or this and pass train_parametes to the optimizer
    # if freeze:
    #     train_parametes = model.collect_params(model.name+"_out*")
    # else:
    #     train_parametes = model.collect_params()

    if learner.lower() == "adagrad":
        trainer = gluon.Trainer(model.collect_params(), 'AdaGrad', {'learning_rate': lr, 'wd': l2reg})
    elif learner.lower() == "rmsprop":
        trainer = gluon.Trainer(model.collect_params(), 'RMSProp', {'learning_rate': lr, 'wd': l2reg})
    elif learner.lower() == "adam":
        trainer = gluon.Trainer(model.collect_params(), 'Adam', {'learning_rate': lr, 'wd': l2reg})
    else:
        trainer = gluon.Trainer(model.collect_params(), 'SGD', {'learning_rate': lr, 'wd': l2reg})
    criterion = gluon.loss.SigmoidBCELoss(from_sigmoid=False)

    # for param in model.collect_params().values():
    #     print (param.name,", ", param.grad_req)

    best_hr, best_ndcgm, best_iter=0,0,0
    for epoch in range(1,epochs+1):
        t1 = time()
        train(model, criterion, trainer, epoch, batch_size, ctx,
            trainRatings,n_items,n_neg,testNegatives)
        t2 = time()
        if epoch % validate_every == 0:
            (hr, ndcg) = evaluate(model, test_loader, ctx, topK)
            print("Epoch: {} {:.2f}s, HR = {:.4f}, NDCG = {:.4f}, validated in {:.2f}s".
                format(epoch, t2-t1, hr, ndcg, time()-t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter, train_time = hr, ndcg, epoch, t2-t1
                if save_model:
                    checkpoint(model, modelpath)

    print("End. Best Iteration {}:  HR = {:.4f}, NDCG = {:.4f}. ".format(best_iter, best_hr, best_ndcg))
    if save_model:
        print("The best NeuMF model is saved to {}".format(modelpath))

    if save_model:
        if not os.path.isfile(resultsdfpath):
            results_df = pd.DataFrame(columns = ["modelname", "best_hr", "best_ndcg", "best_iter",
                "train_time"])
            experiment_df = pd.DataFrame([[modelfname, best_hr, best_ndcg, best_iter, train_time]],
                columns = ["modelname", "best_hr", "best_ndcg", "best_iter","train_time"])
            results_df = results_df.append(experiment_df, ignore_index=True)
            results_df.to_pickle(resultsdfpath)
        else:
            results_df = pd.read_pickle(resultsdfpath)
            experiment_df = pd.DataFrame([[modelfname, best_hr, best_ndcg, best_iter, train_time]],
                columns = ["modelname", "best_hr", "best_ndcg", "best_iter","train_time"])
            results_df = results_df.append(experiment_df, ignore_index=True)
            results_df.to_pickle(resultsdfpath)
