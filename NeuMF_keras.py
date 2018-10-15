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
import argparse

from keras import initializers
from keras.models import Model, load_model, save_model
from keras.layers import Dense, Embedding, Input, Dropout, Flatten, concatenate, multiply
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.regularizers import l2

from time import time
from evaluate_keras import evaluate_model
from Dataset import Dataset
from utils import *
from GMF_keras import GMF
from MLP_keras import MLP

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
    parser.add_argument("--reg_mf", type=float, default=0.,
        help="l2 regularization for the GMF part.")

    # MLP set up
    parser.add_argument("--layers", type=str, default="[64,32,16,8]",
        help="layer architecture. The first elements is used for the embedding \
        layers for the MLP part and equals n_emb*2")
    parser.add_argument("--reg_mlp", type=float, default=0.,
        help="l2 regularization for the MLP part.")
    parser.add_argument("--dropouts", type=str, default="[0.,0.,0.]",
        help="dropout per dense layer. len(dropouts) = len(layers)-1")

    # Output layer
    parser.add_argument("--reg_out", type=float, default=0.,
        help="l2 regularization for the output layer.")

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
    parser.add_argument("--save_model", type=int, default=1)
    parser.add_argument("--n_neg", type=int, default=4,
        help="number of negative instances to consider per positive instance.")
    parser.add_argument("--topK", type=int, default=10,
        help="number of items to retrieve for recommendation.")

    return parser.parse_args()


def NeuMF(n_users, n_items, n_emb, layers, dropouts, reg_mf, reg_mlp, reg_out):

    num_layer = len(layers) #Number of layers in the MLP

    user = Input(shape=(1,), dtype='int32', name = 'user_input')
    item = Input(shape=(1,), dtype='int32', name = 'item_input')

    # user and item embeddings
    MF_Embedding_User = Embedding(input_dim = n_users, output_dim = n_emb,
        name = 'mf_user_embedding', embeddings_initializer='normal',
        embeddings_regularizer=l2(reg_mf), input_length=1)
    MF_Embedding_Item = Embedding(input_dim = n_items, output_dim = n_emb,
        name = 'mf_item_embedding', embeddings_initializer='normal',
        embeddings_regularizer=l2(reg_mf), input_length=1)

    MLP_Embedding_User = Embedding(input_dim = n_users, output_dim = int(layers[0]/2),
        name = 'mlp_user_embedding', embeddings_initializer='normal',
        embeddings_regularizer=l2(reg_mlp), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = n_items, output_dim = int(layers[0]/2),
        name = 'mlp_item_embedding', embeddings_initializer='normal',
        embeddings_regularizer=l2(reg_mlp), input_length=1)

    # GMF part
    mf_user_latent = Flatten()(MF_Embedding_User(user))
    mf_item_latent = Flatten()(MF_Embedding_Item(item))
    mf_vector = multiply([mf_user_latent, mf_item_latent])

    # MLP part
    mlp_user_latent = Flatten()(MLP_Embedding_User(user))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item))
    mlp_vector = concatenate([mlp_user_latent, mlp_item_latent])
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], activation="relu", kernel_regularizer=l2(reg_mlp), name = "layer{}".format(idx))
        mlp_vector = layer(mlp_vector)
        mlp_vector = Dropout(dropouts[idx-1])(mlp_vector)

    predict_vector = concatenate([mf_vector, mlp_vector])

    # Final prediction layer
    prediction = Dense(1, activation='sigmoid',
        kernel_regularizer=l2(reg_out),
        kernel_initializer='lecun_uniform',
        name = 'prediction')(predict_vector)

    # Model
    model = Model(inputs=[user, item], outputs=prediction)

    return model


def load_pretrain_model(model, gmf_model, mlp_model, num_layers):

    # MF embeddings
    gmf_user_embeddings = gmf_model.get_layer('user_embedding').get_weights()
    gmf_item_embeddings = gmf_model.get_layer('item_embedding').get_weights()
    model.get_layer('mf_user_embedding').set_weights(gmf_user_embeddings)
    model.get_layer('mf_item_embedding').set_weights(gmf_item_embeddings)

    # MLP embeddings
    mlp_user_embeddings = mlp_model.get_layer('user_embedding').get_weights()
    mlp_item_embeddings = mlp_model.get_layer('item_embedding').get_weights()
    model.get_layer('mlp_user_embedding').set_weights(mlp_user_embeddings)
    model.get_layer('mlp_item_embedding').set_weights(mlp_item_embeddings)

    # MLP layers
    for i in range(1, num_layers):
        mlp_layer_weights = mlp_model.get_layer("layer{}".format(i)).get_weights()
        model.get_layer("layer{}".format(i)).set_weights(mlp_layer_weights)

    # Prediction weights
    gmf_prediction = gmf_model.get_layer('prediction').get_weights()
    mlp_prediction = mlp_model.get_layer('prediction').get_weights()
    new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0]), axis=0)
    new_b = gmf_prediction[1] + mlp_prediction[1]
    model.get_layer('prediction').set_weights([0.5*new_weights, 0.5*new_b])

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
    reg_mf = args.reg_mf

    layers = eval(args.layers)
    reg_mlp = args.reg_mlp
    dropouts = eval(args.dropouts)

    reg_out = args.reg_out

    freeze = bool(args.freeze)
    mf_pretrain = os.path.join(modeldir, args.mf_pretrain)
    mlp_pretrain = os.path.join(modeldir, args.mlp_pretrain)
    with_pretrained = "wpret" if os.path.isfile(mf_pretrain) else "wopret"
    is_frozen = "frozen" if freeze else "trainable"

    validate_every = args.validate_every
    save_model = bool(args.save_model)
    n_neg = args.n_neg
    topK = args.topK

    modelfname = "keras_NeuMF" + \
        "_" + with_pretrained + \
        "_" + is_frozen + \
        "_" + learner + \
        ".h5"
    modelpath = os.path.join(modeldir, modelfname)
    resultsdfpath = os.path.join(modeldir, 'results_df.p')

    # Loading data
    dataset = Dataset(os.path.join(datadir, dataname))
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    n_users, n_items = train.shape

    # Build model
    model = NeuMF(n_users, n_items, n_emb, layers, dropouts, reg_mf, reg_mlp, reg_out)
    if freeze:
        for layer in model.layers[:-2]:
            layer.trainable = False
    if learner.lower() == "adagrad":
        model.compile(optimizer=Adagrad(lr=lr), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=lr), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=lr), loss='binary_crossentropy')

    # Load pretrain model
    if os.path.isfile(mf_pretrain) and os.path.isfile(mlp_pretrain):
        gmf_model = GMF(n_users, n_items, n_emb, reg_mf)
        gmf_model.load_weights(mf_pretrain)
        mlp_model = MLP(n_users, n_items, layers, dropouts, reg_mlp)
        mlp_model.load_weights(mlp_pretrain)
        model = load_pretrain_model(model, gmf_model, mlp_model, len(layers))
        print("Load pretrained GMF {} and MLP {} models done. ".format(mf_pretrain, mlp_pretrain))

    best_hr, best_ndcg, best_iter = 0,0,0
    for epoch in range(1,epochs+1):
        t1 = time()
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
        print("The best NeuCF model is saved to {}".format(modelpath))

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
