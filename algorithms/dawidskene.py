#!/usr/bin/python
from random import Random
from copy import deepcopy
from algorithms.algointerface import CrowdAlgorithm
from algorithms.majorityvote import MajorityVoting

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
    def __init__(self, full, train, validation, test, init_seed, init_type = "random", max_iter = 100):
        CrowdAlgorithm.__init__(self, "DawidSkene", full, train, validation, test)
        # init classes
        for i in full.questions.values(): # == answers
            if i != None and not i in self.classes:
                self.classes.append(i)
        
        self.C = len(self.classes)

        # init PRNG
        self.rng = Random(init_seed)

        # init param
        self.init = init_type
        self.max_iter = max_iter
    #end constructor

    #@OVERRIDE
    def run(self):
        p_e = self.ds_init()

        # main loop
        for _ in range(0, self.max_iter):
            # Step 1 [semi-supervision] : update p_e with train values
            for q,a in self.train.questions.items():
                for c in self.classes:
                    p_e[q][c] = 1.0 if c == a else 0.0
                #end for
            #end for

            # Step 2 [training] : perform M-step and E-step
            (priors, confusion) = self.M_step(p_e, self.full)
            p_e = self.E_step(priors, confusion, self.full)

            print(priors)

            # Step 3 [validation] : check accuracy on validation and update the matrices
            _, _, f1score, _ = self.validate(self.validation, self.infer_answers(p_e, self.validation))
            print("DEBUG", ("best_f1", self.bestf1), ("current_f1", f1score))
            if f1score >= self.bestf1:
                self.bestf1 = f1score
                (self.priors, self.confusion) = (priors, confusion)
            #end if

            # Step 4 [convergence check] : check if our iterations don't bring value any more
            if self.convergence(priors, confusion, p_e):
                break
        #end while

        # return testset evaluated on best valid
        p_e = self.E_step(self.priors, self.confusion, self.test)
        return self.infer_answers(p_e, self.test)
    #end function

    ###########################
    ### ORGANIZATIONAL PART ###
    ###########################

    def infer_answers(self, p_e, dataset):
        answers = {}
        for q in dataset.questions.keys():
            prob = p_e[q]
            ans = self.idx_max(prob)
            answers[q] = ans
        return answers


    def idx_max(self,arr):
        val = -1
        i = -1

        for e in range(0, len(arr)):
            if arr[e] > val:
                val = arr[e]
                i = e

        return i


    ########################
    ### ALGORITHMIC PART ###
    ########################

    def convergence(self, priors, confusion, p_e):
        # pull up data from cache
        (last_priors, last_confusion, last_p_e) = (self.last_priors, self.last_confusion, self.last_p_e)

        # check for convergence
        converged = False

        #todo: implement

        # update cached values
        (self.last_priors, self.last_confusion, self.last_p_e) = (priors, confusion, p_e)

        # return result
        return converged

    def ds_init(self):
        # init p_e with either mv (2 types), or at random (2 types)
        if self.init == "random":
            print("[DS::Init] random_init")
            p_e = dict([(q, [0.0 for c in self.classes]) for q,_ in self.full.questions.items()]) # 0-init first
            
            for q in self.full.questions.keys():
                p_e[q][self.classes[self.rng.randint(0, self.C - 1)]] = 1.0 # set a random class to 1
        elif self.init == "flat":
            print("[DS::Init] flatbase_random_init")
            p_e = dict([(q, [1.0 / len(self.classes) for c in self.classes]) for q,_ in self.full.questions.items()])

            for q in self.full.questions.keys():
                mod = self.rng.uniform(-0.5,0.5) / 3
                for c in self.classes:
                    p_e[q][c] += ((-1) ** (c+1)) * mod
        elif self.init == "mv":
            print("[DS::Init] majority_voting")
            p_e = dict([(q, [0.0 for c in self.classes]) for q,_ in self.full.questions.items()])
            mv_answers = MajorityVoting(None, None, None, self.full).run()

            for q,a in mv_answers.items():
                for c in self.classes:
                    p_e[q][c] = 1.0 if a == c else 0.0
        elif self.init == "mv_w":
            print("[DS::Init] weighted_majority_voting")
            p_e = dict([(q, [0.0 for c in self.classes]) for q,_ in self.full.questions.items()])

            for q,a in self.full.questions.items():
                for c in self.classes:
                    p_e[q][c] = sum([1.0 if a == c else 0.0 for a in self.full.answers])
        else:
            print("[DS::Init::fallback] ground_init")
            print("Unsupported init for semi-supervised, aborting")
            exit()
            p_e = dict([(q, [1.0 if c == a else 0.0 for c in self.classes]) for q,a in self.full.questions.items()])

        return p_e


    def E_step(self, priors, confusion, dataset):
        p_e = dict()

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