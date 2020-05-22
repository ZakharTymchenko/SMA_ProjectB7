#!/usr/bin/python
import time
from random import Random
import copy
from copy import deepcopy
from algorithms.algointerface import CrowdAlgorithm
import numpy as np

class DawidSkene(CrowdAlgorithm):
    # data info
    classes = []
    rng = None
    priors, confusion = None, None
    bestf1 = -1.0
    last_p_e, last_priors, last_confusion = None, None, None

    # params
    init = "random"
    max_iter = 100

    """
    The constructor for Dawid-Skene crowd algorithm.
    ArgIn:
        -- full : DATASET with all questions, workers and answers
        -- train, validation, test : DATSET with limited questions and resp. subsets of workers and answers
                                     test should have ground answers removed and set to None
        -- init_type : String, determines init type for p_e. Options are:
            ** random : random 1 or 0 for each class;
            ** flat : 0.5 +/- rnd for each class;
            ** mv : winner-take-all majority voting;
            ** mv_w : p_e gets assigned a probability with all votes having equal count
        -- max_iter : Int, sets an upper limit to the # of iterations of DS that can't be breached regardless of convergence
    No training is performed in the constructor.
    """

    def __init__(self, full, train, validation, test, init_seed, init_type="random", max_iter=100):
        CrowdAlgorithm.__init__(self, "DawidSkene", full, train, validation, test)
        # init classes
        for i in full.questions.values():  # == answers
            if i != None and not i in self.classes:
                self.classes.append(i)

        self.C = len(self.classes)

        # init PRNG
        self.rng = Random(init_seed)

        # init param
        self.init = init_type
        self.max_iter = max_iter

    # end constructor

    #@OVERRIDE
    def run(self):
        c = self.C

        self.init_qorder(self.full)
        self.calc_t_nj_w(self.full)

        p_e = self.random_init(self.full)
        avg_time = 0
        ##DEBUG
        for iter in range(0, self.max_iter):
            t0 = time.time()
            print('Iteration' + str(iter))
            # Step 1 [semi-supervision] : update p_e with train values
            for q,a in self.train.questions.items():
                for c in range(self.C):
                     p_e[c, self.q_order[q]] = 1.0 if c == a else 0.0
                #end for
            #end for

            # Step 2 [training] : perform M-step and E-step
            p_i, conf_mat_w = self.m_step(p_e, self.full)
            p_e = self.e_step(p_i, conf_mat_w, self.full)



            # Step 3 [validation] : check accuracy on validation and update the matrices
            _, _, f1score, _ = self.validate(self.validation, self.infer_answers(p_e, self.validation))
            if f1score >= self.bestf1:
                self.bestf1 = f1score
                (self.priors, self.confusion) = (p_i, conf_mat_w)
            #end if

            # Step 4 [convergence check] : check if our iterations don't bring value any more
            if self.convergence(p_i, conf_mat_w, p_e):
                break
            t1 = time.time()
            avg_time += t1-t0

        if iter != 0:
            avg_time /= iter

        print('Average time per iteration is: ' + str(avg_time) + 's')
        self.init_qorder(self.test)
        self.calc_t_nj_w(self.test)
        p_e = self.e_step(p_i, conf_mat_w, self.test)
        return self.infer_answers(p_e, self.test)
    #end function


    def convergence(self, priors, confusion, p_e):
        # pull up data from cache
        (last_priors, last_confusion, last_p_e) = (self.last_priors, self.last_confusion, self.last_p_e)

        # check for convergence
        converged = False

        # check if this is first iteration, if yes convergence check can't be done
        if last_p_e is not None:
            # check one after another if p_e, priors and confusion matrices are converged
            if np.array_equal(np.round(p_e, 1), np.round(last_p_e, 1)) and \
                    np.array_equal(np.round(priors, 1), np.round(last_priors, 1)) and\
                    np.array_equal(np.round(confusion, 1), np.round(last_confusion, 1)):
                        converged = True

        # update cached values
        self.last_p_e = copy.deepcopy(p_e)
        (self.last_priors, self.last_confusion) = (priors, confusion)

        # return result
        return converged


    ###########################
    ### ORGANIZATIONAL PART ###
    ###########################

    def infer_answers(self, p_e, dataset):
        answers = {}
        i = 0
        for q in dataset.questions.keys():

            ans = np.argmax(p_e[:,i])
            answers[q] = ans
            i += 1

        return answers



    def m_step(self, true_labels, trainset):

        c = self.C
        # Estimate the class prior

        # number of itemns
        n = len(trainset.questions.keys())

        p_i = np.sum(true_labels,  axis=1) / n

        t_nj_w = self.t_nj_w

        conf_mat_w = np.zeros([len(trainset.workers), c, c])

        for w_count in range(len(trainset.workers)):
            for i in range(c):
                for j in range(c):
                    conf_mat_w[w_count, i, j] = np.dot(true_labels[i, :], t_nj_w[w_count, j, :])

                # divide the row through the sum of
                conf_bot = np.sum(conf_mat_w[w_count, i, :])
                if conf_bot > 0:
                    conf_mat_w[w_count, i, :] = conf_mat_w[w_count, i, :] / conf_bot


        return (p_i, conf_mat_w)

    def e_step(self, p_i, conf_mat_w, trainset):
        c = self.C
        n = len(trainset.questions)

        p_e = np.zeros([c, n])

        conf_mat_w = np.array(conf_mat_w)
        t_nj_w = np.array(self.t_nj_w)

        for i in range(n):
            for j in range(c):
                p_et = p_i[j]
                p_et *= np.prod(np.power(conf_mat_w[:, j, :], t_nj_w[:, :, i]))

                p_e[j, i] = p_et

            p_et_ave = np.sum(p_e[:, i])

            if p_et_ave > 0:
                p_e[:, i] = p_e[:, i] / p_et_ave

        return p_e

    def random_init(self, trainset):

        p_e = np.zeros([self.C, len(trainset.questions.keys())])

        for a in range(0, len(self.full.questions.keys())):
            p_e[self.rng.randint(0, self.C - 1)][a] = 1

        return p_e

    def init_qorder(self,dataset):
        self.q_order = {}

        # init q_order
        i = 0
        for q in dataset.questions.keys():
            self.q_order[q] = i
            i += 1


    def calc_t_nj_w(self, dataset):
        c = self.C
        n = len(dataset.questions)

        t_nj_w = []
        t_nj = []
        for _ in range(0, c):
            t_nj.append([0] * n)
        t_nj_blank = deepcopy(t_nj)
        # create t_nj for every worker, an list that has c rows and n column
        # An entry is 1 if the worker said the item is labeled as c
        for w in dataset.workers:
            for x in dataset.workers[w]:
                j = self.q_order[x[0]]
                t_nj[x[1]][j] = 1

            t_nj_w.append(t_nj)
            t_nj = deepcopy(t_nj_blank)

        self.t_nj_w = np.array(t_nj_w)

    #@OVERRIDE
    def test(self, testset):
        self.init_qorder(testset)
        self.calc_t_nj_w(testset)

        p_e = self.e_step(self.p_i, self.conf_mat_w, self.t_nj_w, testset)
        
        for _ in range(0, 8):
            p_i, conf_mat_w, t_nj_w = self.m_step(p_e, testset)
            p_e = self.e_step(p_i, conf_mat_w, t_nj_w, testset)
        
        for i in testset.questions.keys():
            ord = self.q_order[i]
            prob = list([i[ord] for i in p_e])
            testset.questions[i] = self.idx_max(prob)
            #print((testset.questions[i], prob))

        return testset.questions
    #end function