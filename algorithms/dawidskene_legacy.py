#!/usr/bin/python
import copy
from random import Random
from copy import deepcopy
from algorithms.algointerface import CrowdAlgorithm
import time

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
        
        p_e = []
        for _ in range(0, c):
            p_e.append([0].copy() * len(self.full.questions.keys()))

        for a in range(0, len(self.full.questions.keys())):
            p_e[self.rng.randint(0, c - 1)][a] = 1
        #for q,a in self.train.questions.items(): #question_id, answer
        #    p_e[a][self.q_order[q]] = 1 # first index - class of the answer, second index - question's ordered index


        ##DEBUG

        for iter in range(0, self.max_iter):
            print('Iteration' + str(iter))
            # Step 1 [semi-supervision] : update p_e with train values
            for q,a in self.train.questions.items():
                for c in range(self.C):
                     p_e[c][self.q_order[q]] = 1.0 if c == a else 0.0
                #end for
            #end for

            # Step 2 [training] : perform M-step and E-step
            p_i, conf_mat_w = self.m_step(p_e, self.full)
            p_e = self.e_step(p_i, conf_mat_w, self.full)

            # Step 3 [validation] : check accuracy on validation and update the matrices
            _, _, f1score, _ = self.validate(self.validation, self.infer_answers(p_e, self.validation))
            print("DEBUG", ("best_f1", self.bestf1), ("current_f1", f1score))
            if f1score >= self.bestf1:
                self.bestf1 = f1score
                (self.priors, self.confusion) = (p_i, conf_mat_w)
            #end if

            # Step 4 [convergence check] : check if our iterations don't bring value any more
            if self.convergence(p_i, conf_mat_w, p_e):
                break


        self.init_qorder(self.test)
        self.calc_t_nj_w(self.test)
        p_e = self.e_step(p_i, conf_mat_w, self.test)
        return self.infer_answers(p_e, self.test)
    #end function

    ###########################
    ### ORGANIZATIONAL PART ###
    ###########################

    def infer_answers(self, p_e, dataset):
        answers = {}
        i = 0
        for q in dataset.questions.keys():
            prob = [p_e[0][i], p_e[1][i]]
            ans = self.idx_max(prob)
            answers[q] = ans
            i += 1
        return answers


    def idx_max(self,arr):
        val = -1
        i = -1

        for e in range(0, len(arr)):
            if arr[e] > val:
                val = arr[e]
                i = e

        return i

    def convergence(self, priors, confusion, p_e):
        # pull up data from cache
        (last_priors, last_confusion, last_p_e) = (self.last_priors, self.last_confusion, self.last_p_e)

        # check for convergence
        converged = False

        # check if this is first iteration, if yes convergence check can't be done
        if last_p_e is not None:
            # check one after another if p_e, priors and confusion matrices are converged
            if self.checkConv_p_e(p_e, last_p_e):
                if self.checkConv_priors(priors, last_priors):
                    if self.checkConv_confusion(confusion, last_confusion):
                        converged = True

        # update cached values
        self.last_p_e = copy.deepcopy(p_e)
        (self.last_priors, self.last_confusion) = (priors, confusion)

        # return result
        return converged

    # checks the convergene of the true labels
    def checkConv_p_e(self, p_e, last_p_e):
        for i in range(len(p_e)):
            for j in range(len(p_e[i])):
                if round(p_e[i][j], 1) != round(last_p_e[i][j], 1):
                    return False
        return True

    def checkConv_priors(self, priors, last_priors):
        for i in range(len(priors)):
            if round(priors[i], 1) != round(last_priors[i], 1):
                return False
        return True

    # checks the convergene of the confusion matrices
    def checkConv_confusion(self, confusion, last_confusion):
        for i in range(len(confusion)):
            for j in range(len(confusion[i])):
                for k in range(len(confusion[i][j])):
                    if round(confusion[i][j][k], 1) != round(last_confusion[i][j][k], 1):
                        return False
        return True

    def m_step(self, true_labels, trainset):

        c = self.C
        # Estimate the class prior

        # number of itemns
        n = len(trainset.questions.keys())

        p_i = [0] * c
        # count occurences of classes from true_labels in list
        for i in range(0, len(true_labels)):
            for x in true_labels[i]:
                p_i[i] += x

        # devide through number of items to get class prior
        for i in range(0, len(p_i)):
            p_i[i] = p_i[i] / n

        print(p_i)
        # Estimate the confusion matrices

        #
        T_ni = deepcopy(true_labels)

        t_nj_w = self.t_nj_w

        # confusion matrix
        conf_mat_w = []
        conf_top = 0.0
        conf_bot = [0.0] * c
        conf_mat = []

        for _ in range(0, c):
            conf_mat.append([0.0] * c)

        conf_mat_blank = deepcopy(conf_mat)
        for w_count in range(0, len(trainset.workers)):
            for i in range(0, c):
                for j in range(0, c):
                    for item in range(0, n):
                        conf_top += T_ni[i][item] * t_nj_w[w_count][j][item]

                    conf_mat[i][j] = conf_top
                    conf_bot[i] += conf_top
                    conf_top = 0.0

            # divide the row through the sum of
            for i in range(0, c):
                for j in range(0, c):
                    if conf_bot[i] > 0.0:
                        conf_mat[i][j] = conf_mat[i][j]  / conf_bot[i]

            conf_bot = [0] * c
            conf_mat_w.append(conf_mat)
            conf_mat = deepcopy(conf_mat_blank)

        return deepcopy(p_i), deepcopy(conf_mat_w)


    def e_step(self, p_i, conf_mat_w, trainset):
        c = self.C
        n = len(trainset.questions)

        p_e= []
        for _ in range(0, c):
            p_e.append([0] * n)
        p_et = 1
        p_et_ave = 0
        t_nj_w = self.t_nj_w
        for a in range(0, n):
            for i in range(0, c):
                p_et = p_i[i]
                for m in range(0, len(trainset.workers)):
                    for j in range(0, c):
                        p_et *= conf_mat_w[m][i][j] ** t_nj_w[m][j][a]
                p_e[i][a] = p_et
                p_et_ave += p_et

            if p_et_ave > 0:
                for x in range(0, c):
                    p_e[x][a] = p_e[x][a] / p_et_ave

            p_et_ave = 0
        return deepcopy(p_e)


    def random_init(self,trainset):


        mat = []
        mat_tmp = [-1] * (len(trainset.questions))

        for w in trainset.workers:
            for x in trainset.workers[w]:
                j = self.q_order[x[0]]
                mat_tmp[j] = x[1]
            mat.append(mat_tmp)
            mat_tmp = [-1] * (len(trainset.questions))

        return mat

    def init_qorder(self,dataset):
        self.q_order = {}

        # init q_order
        i = 0
        for q in dataset.questions.keys():
            self.q_order[q] = i
            i += 1

    def idx_max(self,arr):
        val = -1
        i = -1

        for e in range(0, len(arr)):
            if arr[e] > val:
                val = arr[e]
                i = e

        return i

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

        self.t_nj_w = t_nj_w

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