"""
Apart from the neccesary adaptations to compare the results between keras,
pytorch and gluon, and to adapt to keras 2.2, I have tried to leave the code
as similar as possible to the original here:

https://github.com/hexiangnan/neural_collaborative_filtering

All credit for the code here to Xiangnan He and collaborators
"""

import numpy as np
import pandas as pd
import os
import heapq
import keras
import multiprocessing as mp
import argparse

from keras import initializers
from keras.models import Model, load_model, save_model
from keras.layers import Dense, Embedding, Input, merge, Flatten, multiply
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.regularizers import l2

from time import time
from evaluate_keras import evaluate_model
from Dataset import Dataset
from utils import *


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
    parser.add_argument("--reg", type=float, default=0.0,
        help="l2 regularzation")
    parser.add_argument("--n_emb", type=int, default=8,
        help="embedding size.")
    parser.add_argument("--lr", type=float, default=0.01,
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


def GMF(n_users, n_items, n_emb, reg):

    user = Input(shape=(1,), dtype='int32', name = 'user_input')
    item = Input(shape=(1,), dtype='int32', name = 'item_input')

    # user and item embeddings
    MF_Embedding_User = Embedding(
        input_dim = n_users,
        output_dim = n_emb,
        name = 'user_embedding',
        embeddings_initializer='normal',
        embeddings_regularizer=l2(reg),
        input_length=1)
    MF_Embedding_Item = Embedding(
        input_dim = n_items,
        output_dim = n_emb,
        name = 'item_embedding',
        embeddings_initializer='normal',
        embeddings_regularizer=l2(reg),
        input_length=1)

    # Flatten and multiply
    user_latent = Flatten()(MF_Embedding_User(user))
    item_latent = Flatten()(MF_Embedding_Item(item))
    predict_vector = multiply([user_latent, item_latent])

    # output layer
    prediction = Dense(1, activation='sigmoid',
        kernel_regularizer=l2(reg),
        kernel_initializer='lecun_uniform',
        name = 'prediction')(predict_vector)

    # Model
    model = Model(inputs=[user, item], outputs=prediction)

    return model


if __name__ == '__main__':

    args = parse_args()

    datadir = args.datadir
    dataname = args.dataname
    modeldir = args.modeldir
    n_emb = args.n_emb
    reg = args.reg
    batch_size = args.batch_size
    epochs = args.epochs
    learner = args.learner
    lr = args.lr
    validate_every = args.validate_every
    save_model = args.save_model
    topK = args.topK
    n_neg = args.n_neg

    modelfname = "keras_GMF" + \
        "_".join(["_bs", str(batch_size)]) + \
        "_".join(["_reg", str(reg).replace(".", "")]) + \
        "_".join(["_lr", str(lr).replace(".", "")]) + \
        "_".join(["_n_emb", str(n_emb)]) + ".h5"
    modelpath = os.path.join(modeldir, modelfname)
    resultsdfpath = os.path.join(modeldir, 'results_df.p')

    dataset = Dataset(os.path.join(datadir, dataname))
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    n_users, n_items = train.shape

    model = GMF(n_users, n_items, n_emb, reg)
    if learner.lower() == "adagrad":
        model.compile(optimizer=Adagrad(lr=lr), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=lr), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=lr), loss='binary_crossentropy')

    best_hr, best_ndcg, best_iter = 0,0,0
    for epoch in range(1,epochs+1):
        t1 = time()
        # in every iteration we "pull" a new set of negative instances
        user, item, labels = get_train_instances(train, n_items, n_neg, testNegatives)
        hist = model.fit([user, item], labels, batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time()
        if epoch % validate_every ==0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print("Iteration {}: {:.2f}s, HR = {:.4f}, NDCG = {:.4f}, loss = {:.4f}, validated in {:.2f}s"
                .format(epoch, t2-t1, hr, ndcg, loss, time()-t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter, train_time = hr, ndcg, epoch, t2-t1
                if save_model:
                    model.save_weights(modelpath, overwrite=True)

    print("End. Best Iteration {}:  HR = {:.4f}, NDCG = {:.4f}. "
        .format(best_iter, best_hr, best_ndcg))
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
