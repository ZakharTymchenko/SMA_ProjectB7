#!/usr/bin/python
from random import Random
from copy import deepcopy
from algorithms.algointerface import CrowdAlgorithm

class DawidSkene(CrowdAlgorithm):
    # prior
    classes = []
    C = 0
    rng = None
    q_order = {}

    # training
    p_i = None
    conf_mat_w = None
    t_nj_w = None

    def __init__(self, train, validation, init_seed):
        CrowdAlgorithm.__init__(self, "DawidSkene", train, validation)
        # init classes
        for i in train.questions.values(): # == answers
            if not i in self.classes:
                self.classes.append(i)
        
        self.C = len(self.classes)

        # init PRNG
        self.rng = Random(init_seed)
    #end constructor


    #@OVERRIDE
    def fit(self):
        c = self.C

        self.init_qorder(self.train)
        self.calc_t_nj_w(self.train)
        
        p_e = []
        for _ in range(0, c):
            p_e.append([0].copy() * len(self.train.questions.keys()))

        for a in range(0, len(self.train.questions.keys())):
            p_e[self.rng.randint(0, c - 1)][a] = 1
        #for q,a in self.train.questions.items(): #question_id, answer
        #    p_e[a][self.q_order[q]] = 1 # first index - class of the answer, second index - question's ordered index
        
        # M-step single iteration that will infer prior and confusion
        (self.p_i, self.conf_mat_w, self.t_nj_w) = self.m_step(p_e, self.train)

        ##DEBUG
        for _ in range(0, 10):
            p_i, conf_mat_w, t_nj_w = self.m_step(p_e, self.train)
            p_e = self.e_step(p_i, conf_mat_w, t_nj_w, self.train)
        #p_e = self.e_step(self.p_i, self.conf_mat_w, self.t_nj_w, self.train)
        
        acc = 0.0
        total = 0.0
        for i in self.train.questions.keys():
            ord = self.q_order[i]
            prob = list([i[ord] for i in p_e])
            print(i, self.train.questions[i], self.idx_max(prob), prob)
            acc += 1 if self.train.questions[i] == self.idx_max(prob) else 0
            total += 1
        print("acc", acc/total)
        exit()
    #end function

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

        return p_i, conf_mat_w, t_nj_w
    # end function

    def e_step(self, p_i, conf_mat_w, t_nj_w, trainset):
        c = self.C
        n = len(trainset.questions)

        p_e= []
        for _ in range(0, c):
            p_e.append([0] * n)
        p_et = 1
        p_et_ave = 0

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
        return p_e

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

    def calc_t_nj_w(self,dataset):
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