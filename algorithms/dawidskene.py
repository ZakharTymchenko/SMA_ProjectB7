#!/usr/bin/python
import random
from copy import deepcopy
from algorithms.algointerface import CrowdAlgorithm

class DawidSkene(CrowdAlgorithm):
    q_order = {}

    def __init__(self, train, validation):
        CrowdAlgorithm.__init__(self, "DawidSkene", train, validation)


    #end constructor


    #@OVERRIDE
    def fit(self, trainset):

        #init_mat = self.random_init(trainset)
        i = 0
        for q in trainset.questions.keys():
            self.q_order[q] = i
            i += 1
        random.seed(1)
        c = 2
        p_e= []
        for l in range(0, c):
            p_e.append([0].copy() * len(trainset.questions.keys()))

        for a in range(0, len(trainset.questions.keys())):
            p_e[random.randint(0, c - 1)][a] = 1


        #p_e = self.test(trainset)

        for i in range(0, 10):
            p_i, conf_mat_w, t_nj_w = self.m_step(p_e, trainset)
            p_e = self.e_step(p_i, conf_mat_w, t_nj_w, trainset)

    #end function

    def m_step(self, true_labels, trainset):

        c = 2
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

        t_nj_w = []
        t_nj = []
        for l in range(0, c):
            t_nj.append([0] * n)
        t_nj_blank = deepcopy(t_nj)
        # create t_nj for every worker, an list that has c rows and n column
        # An entry is 1 if the worker said the item is labeled as c
        for w in trainset.workers:
            for x in trainset.workers[w]:
                j = self.q_order[x[0]]
                t_nj[x[1]][j] = 1

            t_nj_w.append(t_nj)
            t_nj = deepcopy(t_nj_blank)

        # confusion matrix
        conf_mat_w = []
        conf_top = 0
        conf_bot = [0] * c
        conf_mat = [[0]*c] *c

        for l in range(0, c):
            conf_mat.append([0] * c)

        conf_mat_blank = deepcopy(conf_mat)
        for w_count in range(0, len(trainset.workers)):
            for i in range(0, c):
                for j in range(0, c):
                    for item in range(0, n):
                        conf_top += T_ni[i][item] * t_nj_w[w_count][j][item]

                    conf_mat[i][j] = conf_top
                    conf_bot[i] += conf_top
                    conf_top = 0

                # devide the row through the sum of
                for j in range(0, c):
                    if conf_bot[i] == 0:
                        conf_mat[i][j] = 0
                    else:
                        conf_mat[i][j]  = conf_mat[i][j]  / conf_bot[i]
                conf_bot = [0] * c
            conf_mat_w.append(conf_mat)
            conf_mat = deepcopy(conf_mat_blank)

        return p_i, conf_mat_w, t_nj_w
    # end function

    def e_step(self, p_i, conf_mat_w, t_nj_w, trainset):


        c = 2
        # number of itemns
        n = len(trainset.questions)

        p_e= []
        for l in range(0, c):
            p_e.append([0] * n)
        p_et = 1
        p_et_ave = 0

        for a in range(0, n):
            for i in range(0, c):
                p_et = p_i[i]
                for m in range(0, len(trainset.workers)):
                    for j in range(0, c):
                        p_et *= conf_mat_w[m][i][j] ** t_nj_w[m][i][a]
                p_e[i][a] = p_et
                p_et_ave += p_et

            if p_et_ave > 0:
                for x in range(0, c):
                    p_e[x][a] = p_e[x][a] / p_et_ave

            p_et_ave = 0
        print(p_e)
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
    #@OVERRIDE
    def test(self, testset):
        return 0
    #end function