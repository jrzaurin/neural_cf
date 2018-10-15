"""
Apart from the neccesary adapatations to compare the results between keras,
pytorch and gluon, and to adapt to keras 2.2, I have tried to leave the code
as similar as possible to the original here:

https://github.com/hexiangnan/neural_collaborative_filtering

All credit for the code here to Xiangnan He and collaborators
"""

import math
import heapq
import numpy as np


def evaluate_model(model, testRatings, testNegatives, topK):

    global _model
    global _testRatings
    global _testNegatives
    global _topK

    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _topK = topK

    hits, ndcgs = [],[]
    for idx in range(len(_testRatings)):
        (hr,ndcg) = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)
    return (hits, ndcgs)


def eval_one_rating(idx):

    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    map_item_score = {}
    users = np.full(len(items), u, dtype = 'int32')
    predictions = _model.predict([users, np.array(items)],
                                 batch_size=100, verbose=0)

    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()

    ranklist = heapq.nlargest(_topK, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
