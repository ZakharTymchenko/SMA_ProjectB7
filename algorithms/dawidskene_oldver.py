#!/usr/bin/python
from random import Random
from copy import deepcopy
from algorithms.algointerface import CrowdAlgorithm

class DawidSkene(CrowdAlgorithm):
    # data info
    classes = []
    rng = None
    priors, confusion = None, None

    # params
    random_init = "truth"

    def __init__(self, full, train, validation, test, init_seed, random_init = "truth"):
        CrowdAlgorithm.__init__(self, "DawidSkene", full, train, validation, test)
        # init classes
        for i in train.questions.values(): # == answers
            if not i in self.classes:
                self.classes.append(i)
        
        self.C = len(self.classes)

        # init PRNG
        self.rng = Random(init_seed)

        # init param
        self.init = random_init
    #end constructor


    #@OVERRIDE
    def fit(self):
        # init p_e with either training truth, or at random
        if self.init == "random":
            print("[Training] random_init")
            p_e = dict([(q, [0.0 for c in self.classes]) for q,_ in self.train.questions.items()]) # 0-init first
            
            for q in self.train.questions.keys():
                p_e[q][self.classes[self.rng.randint(0, self.C - 1)]] = 1.0 # set a random class to 1
        elif self.init == "flat":
            print("[Training] flatbase_random_init")
            p_e = dict([(q, [1.0 / len(self.classes) for c in self.classes]) for q,_ in self.train.questions.items()])

            for q in self.train.questions.keys():
                mod = self.rng.uniform(-0.5,0.5) / 3
                for c in self.classes:
                    p_e[q][c] += ((-1) ** (c+1)) * mod
        elif self.init == "mv":
            print("[Training] majority_voring")
            p_e = dict([(q, [0.0 for c in self.classes]) for q,_ in self.train.questions.items()])

            for q in self.train.questions.keys():
                for c in self.classes:
                    p_e[q][c] = sum([1.0 if a == c else 0.0 for a in self.train.answers ])
        else:
            print("[Training] ground_init")
            p_e = dict([(q, [1.0 if c == a else 0.0 for c in self.classes]) for q,a in self.train.questions.items()])
        #end if
        
        ##DEBUG
        for _ in range(0, 1): #set to 1 for a singular iteration
            (priors, confusion) = self.M_step(p_e, self.train)
            p_e = self.E_step(priors, confusion, self.train)
        
        acc = 0.0
        total = 0.0
        for q in self.train.questions.keys():
            prob = p_e[q]#list([prob[q] for prob in p_e])
            ans = self.idx_max(prob)
            print(q, self.train.questions[q], ans, prob)
            acc += 1 if self.train.questions[q] == ans else 0
            total += 1
        print("acc", acc/total)
        #exit()
        self.priors, self.confusion = priors, confusion
    #end function

    def E_step(self, priors, confusion, dataset):
        p_e = dict()
        
        print(priors)
        for q in dataset.questions:
            Qp_e = [0.0 for _ in self.classes]
            total = 0.0

            # only get workers' answers to this question
            workers = [(w, [a[1] for a in ans if a[0] == q]) for w,ans in dataset.workers.items()]
            workers = dict([(w,ans[0]) for w,ans in workers if ans != []])
            
            for i,prob in priors.items():
                for (w,ans) in workers.items():
                    j = ans
                    prob *= confusion[w][i][j] # this will take a p_i as a baseline and multiply it by pi_i,j elements
                #end for
                Qp_e[i] = prob
                total += prob
            #end for
            p_e[q] = [prob / total if total > 0.0 else 1.0/len(self.classes) for prob in Qp_e] # class inferrence impossible, set to random

        return p_e

    def M_step(self, p_e, dataset):
        # step 0 : some vars
        q_no = float(len(dataset.questions))

        # step 1 : prior
        priors = dict([(c, sum([p_e[q][c] for q in dataset.questions.keys()]) / q_no) for c in self.classes])

        # step 2 : confusion
        confusion = dict()

        for w,ans in dataset.workers.items():
            matrix = [[0.0 for _ in self.classes] for _ in self.classes]

            # precalc denominator
            denominator = dict([(c, 0.0) for c in self.classes])

            for (q,_) in ans:
                for i in self.classes:
                    denominator[i] += p_e[q][i]
            
            # calc confusion
            for i in self.classes:
                if denominator[i] == 0.0:
                    matrix[i][j] = 1.0 / len(self.classes) # uniform, his guess is as good as random
                    continue
                for q,j in ans:
                    matrix[i][j] += p_e[q][i] / denominator[i]
            
            confusion[w] = matrix
        
        return (priors, confusion)
    #end function

    def idx_max(self,arr):
        val = -1
        i = -1

        for e in range(0, len(arr)):
            if arr[e] > val:
                val = arr[e]
                i = e
        
        return i

    #@OVERRIDE
    def test(self, testset):
        p_e = self.E_step(self.priors, self.confusion, testset)
        
        for _ in range(0, 100):
            (priors, confusion) = self.M_step(p_e, testset)
            p_e = self.E_step(priors, confusion, testset)
        
        for q in testset.questions.keys():
            testset.questions[q] = self.idx_max(p_e[q])

        return testset.questions
    #end function