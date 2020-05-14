#!/usr/bin/python
from algorithms.datastructure import Dataset
"""
Inherit with:
    class MyAlgorithm(CrowdAlgorithm):
        ...

"""
class CrowdAlgorithm:
    # algorithm name
    name = ""

    full = Dataset()
    train = Dataset()
    validation = Dataset()
    test = Dataset()

    """
    This constructor is called from the subclass with:
        CrowdAlgorithm.__init__(self, "MyName", ...)
    Then all fields that subclass might need in addition are init'd
    No training is performed in the constructor.
    """
    def __init__(self, name, full, train, validation, test):
        self.name = name
        self.full = full
        self.train = train
        self.validation = validation
        self.test = test
    #end constructor

    """
    This function is overriden in the subclass with the same signature
    ArgIn:
        -- testset : dict with KEY=QUESTION, VALUE=None
    """
    def run(self):
        print("called abstract function run/0 in CrowdAlgorithm")
        return dict()
    #end function

    """
    This function is NOT to be overriden in the subclass.
    Dataset should contain _reference_ values and answers should contain inferred answers.
    Binary classification only.
    ArgIn:
        -- dataset : is DATASET
        -- answers : dict with KEY=QUESTION, VALUE=ANSWER
    Returns:
        -- precision, recall, f1score : are float64, representing accuracy metrics resp. to their name
    """

    def validate(self, dataset, answers):
        true_pos = 0.0
        true_neg = 0.0

        false_pos = 0.0
        false_neg = 0.0
        for q in dataset.questions.keys():
            ans = answers[q]
            ref = dataset.questions[q]
            if ans == 1 and ref == 1:
                true_pos += 1.0
            elif ans == 1 and ref == 0:
                false_pos += 1.0
            elif ans == 0 and ref == 0:
                true_neg += 1.0
            elif ans == 0 and ref == 1:
                false_neg += 1.0
            else:
                print("classification is not binary, aborting")
                exit()
        #end for
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        f1score = 2 * (precision * recall) / (precision + recall)
        accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
        return (precision, recall, f1score, accuracy)
    #end function