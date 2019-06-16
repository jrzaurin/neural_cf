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

from Dataset import Dataset as ml1mDataset
from time import time
from utils import *

import pdb

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--datadir", type=str, default="Data_Javier/",
        help="data directory.")
    parser.add_argument("--modeldir", type=str, default="models",
        help="models directory")
    parser.add_argument("--dataname", type=str, default="ml-1m",
        help="chose a dataset.")
    parser.add_argument("--epochs", type=int, default=20,
        help="number of epochs.")
    parser.add_argument("--batch_size", type=int, default=256,
        help="batch size.")
    parser.add_argument("--n_emb", type=int, default=8,
        help="embedding size.")
    parser.add_argument("--lr", type=float, default=0.001,
        help="learning rate.")
    parser.add_argument("--learner", type=str, default="adam",
        help="Specify an optimizer: adagrad, adam, rmsprop, sgd")
    parser.add_argument("--validate_every", type=int, default=1,
        help="validate every n epochs")
    parser.add_argument("--save_model", type=int, default=1)
    parser.add_argument("--n_neg", type=int, default=4,
        help="number of negative instances to consider per positive instance.")
    parser.add_argument("--topK", type=int, default=10,
        help="number of items to retrieve for recommendation.")

    return parser.parse_args()


class GMF(Block):
    """
    General Matrix factorization (without bias)
    """
    def __init__(self, n_user, n_item, n_emb=8):
        super(GMF, self).__init__()

        self.n_emb = n_emb
        self.n_user = n_user
        self.n_item = n_item
        with self.name_scope():
            self.embeddings_user = nn.Embedding(n_user, n_emb, weight_initializer='normal')
            self.embeddings_item = nn.Embedding(n_item, n_emb, weight_initializer='normal')
            # self.out = nn.Dense(in_units=n_emb, units=1, activation='sigmoid', weight_initializer='uniform')
            self.out = nn.Dense(in_units=n_emb, units=1, weight_initializer='uniform')

    def forward(self, users, items):

        user_emb = self.embeddings_user(users)
        item_emb = self.embeddings_item(items)
        prod = user_emb*item_emb
        preds = self.out(prod)

        return preds


# Train and test functions: very similar to pytorch syntax
def train(model, criterion, trainer, epoch, batch_size, ctx,
    trainRatings,n_items,n_neg,testNegatives):
    train_obs = get_train_instances(trainRatings,
        n_items,
        n_neg,
        testNegatives,
        mode="gluon")
    train_dataset = mx.gluon.data.dataset.ArrayDataset(train_obs)
    train_data_loader = mx.gluon.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        # num_workers=4,
        shuffle=True)
    train_steps = (train_obs.shape[0]//batch_size)+1
    running_loss=0
    for batch_idx, data in enumerate(train_data_loader):
        data = data.as_in_context(ctx)
        users, items, labels = data[:,0], data[:,1], data[:,2]
        with autograd.record():
            output = model(users, items)
            loss = criterion(output, labels.astype('float32'))
        loss.backward()
        trainer.step(batch_size)
        running_loss += loss.asnumpy().mean()
    return running_loss/train_steps


def evaluate(model, test_loader, ctx, topK):
    hits, ndcgs = [],[]
    for batch_idx, data in enumerate(test_loader):
        data = data.as_in_context(ctx)
        users, items, labels = data[:,0], data[:,1], data[:,2]
        preds = model(users, items)

        items = items.asnumpy()
        preds = preds.asnumpy()

        gtItem = items[0]

        # the following 3 lines of code ensure that the fact that the 1st item is
        # gtitem does not affect the final rank
        randidx = np.arange(100)
        np.random.shuffle(randidx)
        items, preds = items[randidx], preds[randidx]

        map_item_score = dict( zip(items, preds) )
        ranklist = heapq.nlargest(topK, map_item_score, key=map_item_score.get)
        hr = getHitRatio(ranklist, gtItem)
        ndcg = getNDCG(ranklist, gtItem)
        hits.append(hr)
        ndcgs.append(ndcg)
    return (np.array(hits).mean(),np.array(ndcgs).mean())


def checkpoint(model, modelpath):
    model.save_parameters(modelpath)


if __name__ == '__main__':
    args = parse_args()

    datadir = args.datadir
    dataname = args.dataname
    modeldir = args.modeldir
    n_emb = args.n_emb
    batch_size = args.batch_size
    epochs = args.epochs
    learner = args.learner
    lr = args.lr
    validate_every = args.validate_every
    save_model = args.save_model
    topK = args.topK
    n_neg = args.n_neg

    modelfname = "gluon_GMF" + \
        "_".join(["_bs", str(batch_size)]) + \
        "_".join(["_lr", str(lr).replace(".", "")]) + \
        "_".join(["_n_emb", str(n_emb)]) + ".params"
    modelpath = os.path.join(modeldir, modelfname)
    resultsdfpath = os.path.join(modeldir, 'results_df.p')

    dataset = ml1mDataset(os.path.join(datadir, dataname))
    trainRatings, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    n_users, n_items = trainRatings.shape

    # here we know that we will test using 100 instances and that the 1st one
    # is the positive. Therefore we set batch_size=100 and no shuffle
    test = get_test_instances(testRatings, testNegatives)
    test_dataset = mx.gluon.data.dataset.ArrayDataset(test)
    test_loader = mx.gluon.data.DataLoader(dataset=test_dataset,
        batch_size=100,
        # num_workers=4,
        shuffle=False)

    ctx =  mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
    model = GMF(n_users, n_items, n_emb=n_emb)
    model.initialize(ctx=ctx)
    criterion = gluon.loss.SigmoidBCELoss(from_sigmoid=False)
    if learner.lower() == "adagrad":
        trainer = gluon.Trainer(model.collect_params(), 'AdaGrad', {'learning_rate': lr})
    elif learner.lower() == "rmsprop":
        trainer = gluon.Trainer(model.collect_params(), 'RMSProp', {'learning_rate': lr})
    elif learner.lower() == "adam":
        trainer = gluon.Trainer(model.collect_params(), 'Adam', {'learning_rate': lr})
    else:
        trainer = gluon.Trainer(model.collect_params(), 'SGD', {'learning_rate': lr})

    best_hr, best_ndcgm, best_iter=0,0,0
    for epoch in range(1,epochs+1):
        t1 = time()
        loss = train(model, criterion, trainer, epoch, batch_size, ctx,
            trainRatings,n_items,n_neg,testNegatives)
        t2 = time()
        if epoch % validate_every == 0:
            (hr, ndcg) = evaluate(model, test_loader, ctx, topK)
            print("Iteration {}: {:.2f}s, HR = {:.4f}, NDCG = {:.4f}, loss = {:.4f}, validated in {:.2f}s"
                .format(epoch, t2-t1, hr, ndcg, loss, time()-t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter, train_time = hr, ndcg, epoch, t2-t1
                if save_model:
                    checkpoint(model, modelpath)

    print("End. Best Iteration {}:  HR = {:.4f}, NDCG = {:.4f}. ".format(best_iter, best_hr, best_ndcg))
    if save_model:
        print("The best GMF model is saved to {}".format(modelpath))

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

