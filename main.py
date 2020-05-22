#!/usr/bin/python
from copy import deepcopy

from data.reader import DataReader
from data.splitter import SplitData

from algorithms.datastructure import Dataset
from algorithms.datastructure import ArrangeData

from algorithms.algointerface import CrowdAlgorithm
from algorithms.majorityvote import MajorityVoting
from algorithms.dawidskene_numpy import DawidSkene
#from algorithms.dawidskene_legacy import DawidSkene
#from algorithms.dawidskene import DawidSkene

# this file is the entry point to the project, everything is branched from here

# constants
main_path = "data"
split_seed = 2138796
ds_seed = 9453657


def main():
    # read the data
    reader = DataReader(main_path)

    questions = reader.questions
    workers = reader.workers
    answers = reader.answers

    print("Finished reading answers from " + reader.answer)
    print("Total questions: " + str(len(questions)))
    print("Total workers: " + str(len(workers)))
    print("Total answers: " + str(len(answers)))

    print("class 0:", sum([1 if i == 0 else 0 for i in questions.values()]) / sum([1 for _ in questions.values()]))

    # split into train/validation/test
    (q_train, q_validation, q_test) = SplitData(split_seed, list(questions.items()))
    (full, train, validation, test) = ArrangeData(questions, q_train, q_validation, q_test, workers, answers)

    print("Train size", len(train.questions), len(train.workers), len(train.answers))
    print("Validation size", len(validation.questions), len(validation.workers), len(validation.answers))
    print("Test size", len(test.questions), len(test.workers), len(test.answers))

    # cut ground for test set but keep a local copy
    test_set = deepcopy(test)
    for q in test.questions.keys():
        test.questions[q] = None
        full.questions[q] = None
    #end for

    # initialize algorithms
    algos = []
    algos.append(MajorityVoting(full, train, validation, test))
    algos.append(DawidSkene(full, train, validation, test, ds_seed, "mv_w", 100))

    for alg in algos:
        print("")
        print("Algorithm - " + alg.name)
        # run the algo
        test_ans = alg.run()

        # get back metrics
        (precision, recall, f1score, accuracy) = alg.validate(test_set, test_ans)
        print(("precision", precision))
        print(("recall", recall))
        print(("f1score", f1score))
        print(("accuracy", accuracy))
    #end for
#end function


if __name__ == "__main__":
    main()
#end function