#!/usr/bin/python
from random import Random

def SplitData(split_seed, questions):
    n = len(questions)
    rng = Random(split_seed)
    rng.shuffle(questions)

    train_size = (n * 6) / 10
    valid_size = (n * 2) / 10
    #test_size = (n * 2) / 10 # remainder

    print("Split the dataset using seed " + str(split_seed))

    train = dict()
    valid = dict()
    test = dict()

    for (k,v) in questions:
        if len(train) < train_size:
            train[k] = v
        elif len(valid) < valid_size:
            valid[k] = v
        else:
            test[k] = v
        #end if
    #end for

    return (train, valid, test)