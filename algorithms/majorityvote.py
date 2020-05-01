#!/usr/bin/python
from algorithms.algointerface import CrowdAlgorithm

class MajorityVoting(CrowdAlgorithm):
    def __init__(self, train, validation):
        CrowdAlgorithm.__init__(self, "MajorityVoing", train, validation)
    #end constructor


    #@OVERRIDE
    def fit(self, trainset):
        None
    #end function


    #@OVERRIDE
    def test(self, testset):
        q_res = dict()

        # for each question add an entry QUESTION => [0, 0]
        for q in testset.questions.keys():
            q_res[q] = [0, 0]
        #end for

        # for each answer pick one of the entries from [N1, N2]
        #   and increment it by one, depending on the answer given
        for ans in testset.answers:
            q_res[ans[0]][ans[2]] += 1
        #end for

        # iterate through questions, get an array of answers, find max
        # then replace an entry with estimated answer
        for q in testset.questions.keys():
            cnter = q_res[q]
            ans = 0 if cnter[0] > cnter[1] else 1
            q_res[q] = ans
        #end for

        return q_res
    #end function