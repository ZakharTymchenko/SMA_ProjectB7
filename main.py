#!/usr/bin/python
from data.reader import DataReader
from data.splitter import SplitData

from algorithms.datastructure import Dataset
from algorithms.datastructure import ArrangeData

from algorithms.algointerface import CrowdAlgorithm
from algorithms.majorityvote import MajorityVoting
from algorithms.dawidskene import DawidSkene

# this file is the entry point to the project, everything is branched from here

# constants [todo: rework into CLI args with hardcoded defaults]
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
    (train, validation, test) = ArrangeData(q_train, q_validation, q_test, workers, answers)

    print("Train size", len(train.questions), len(train.workers), len(train.answers))
    print("Validation size", len(validation.questions), len(validation.workers), len(validation.answers))
    print("Test size", len(test.questions), len(test.workers), len(test.answers))

    # initialize algorithms
    algos = []
    algos.append(MajorityVoting(train, validation))
    algos.append(DawidSkene(train, validation, ds_seed))

    for alg in algos:
        print("")
        print("Algorithm - " + alg.name)


        # get a test set w/o ground values
        test_set = Dataset()
        for k in test.questions.keys():
            test_set.questions[k] = None
        #end for
        test_set.workers = test.workers
        test_set.answers = test.answers

        # train the algo
        alg.fit()
        # get back test result
        test_res = alg.test(test_set)

        # process the output
        #todo
        accuracy = 0.0
        for q in test_res:
            if test_res[q] == test.questions[q]:
                accuracy += 1.0
            #end if
        #end for

        accuracy /= len(test.questions)
        print("accuracy", accuracy)
    #end for
#end function


if __name__ == "__main__":
    main()
#end function