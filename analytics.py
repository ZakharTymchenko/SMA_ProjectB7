#!/usr/bin/python
from copy import deepcopy

from data.reader import DataReader
from data.splitter import SplitData

import numpy as np
import statistics as stats
import matplotlib.pyplot as plt

main_path = "data"

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
    print()

    # questions
    print("class 0:", sum([1 if i == 0 else 0 for i in questions.values()]) / sum([1 for _ in questions.values()]))
    print("class 1:", sum([1 if i == 1 else 0 for i in questions.values()]) / sum([1 for _ in questions.values()]))
    print()

    # workers
    w_accuracy = {}
    w_answers = {}
    w_count = {}
    for w,ans in workers.items():
        answers = [0.0,0.0]
        total = 0.0
        correct = 0.0
        
        for (q,a) in ans:
            # accuracy
            total += 1.0
            correct += 1.0 if a == questions[q] else 0.0
            answers[a] += 1.0
        #end for
        w_accuracy[w] = correct / total
        w_answers[w] = answers[1] / (answers[0] + answers[1])
        w_count[w] = total
    #end for

    # values
    print("# of answers per worker")
    print(("mean", stats.mean(w_count.values())), ("stddev", stats.stdev(w_count.values())), ("median", stats.median(w_count.values())))
    print(("min", min(w_count.values())), ("max", max(w_count.values())))
    print()

    w_count_s = sorted(w_count.values())
    count_90 = (len(w_count_s) * 90) / 100
    w_count_s = w_count_s[:int(count_90)]
    print("# of answers per worker (90th percentile)")
    print(("mean", stats.mean(w_count_s)), ("max", max(w_count_s)))
    print()


    print("# of answers per worker")
    print(("mean", stats.mean(w_accuracy.values())), ("stddev", stats.stdev(w_accuracy.values())), ("median", stats.median(w_accuracy.values())))
    print(("min", min(w_accuracy.values())), ("max", max(w_accuracy.values())))
    print()

    #exit()
    # plotting
    plt.hist(w_count.values())
    plt.title("Distribution of worker answer count")
    plt.xlabel("# answers")
    plt.ylabel("# workers")
    plt.show()

    bins = (np.arange(0, 13) / 11.0) - 0.05 #to have areas that'd be centered around 0.0, 0.1, ... 1.0

    plt.hist(w_accuracy.values(), bins=bins)
    plt.title("Distribution of worker accuracy")
    plt.xlabel("% of correct answers")
    plt.ylabel("# workers")
    plt.show()

    plt.hist(w_answers.values(), bins=(bins + 0.02158749248346362)) # to make a prior line be more in the center of a block
    plt.axvline(x=0.12158749248346362, ymin=0, ymax=1, color="red")
    plt.title("Distribution of worker answers")
    plt.xlabel("% of positive answers")
    plt.ylabel("# workers")
    plt.show()

    plt.scatter(w_answers.values(), w_accuracy.values())
    plt.title("Distribution of worker answers to worker accuracy")
    plt.xlabel("% of positive answers")
    plt.ylabel("% of correct answers")
    plt.show()

    plt.scatter(w_count.values(), w_accuracy.values())
    plt.title("Distribution of worker activity to worker accuracy")
    plt.xlabel("# answers")
    plt.ylabel("% of correct answers")
    plt.show()
#end function


if __name__ == "__main__":
    main()
#end function