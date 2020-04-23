#!/usr/bin/python

class DataReader:
    __path = "./"
    __groundfile_base = "files/truth.csv"
    __answerfile_base = "files/answer.csv"

    ground = ""
    answer = ""

    questions = dict()
    workers = dict()
    answers = []

    def __init__(self, datapath = "."):
        self.__path = datapath + ("" if datapath.endswith("/") else "/")
        self.ground = self.__path + self.__groundfile_base
        self.answer = self.__path + self.__answerfile_base

        """
        it seems stupid but despite assigning it, we never use the ground 
        variable because the first entry in answer.csv is of the form 
        ID1_ID2_VAL where VAL is 0 or 1 and is the correct answer... 
        this was verified with the truth file where the format of the 
        first column is the same, and the value in the second one 
        always matches VAL from the first
        """

        #tba
        ans_file = open(self.answer)
        ans_file.next() #skip header
        for i in ans_file:
            line = i.split(',')
            qa = line[0]
            w = line[1]
            w_ans = int(line[2])

            qa_split = qa.split('_')
            (id1, id2, truth) = (int(qa_split[0]), int(qa_split[1]), int(qa_split[2]))
            q = (id1, id2)

            # add the question and its ground to the questions dictionary
            if not q in self.questions:
                self.questions[q] = truth
            #end if

            # add the worker and their answer to the q to workers dict
            if not w in self.workers:
                self.workers[w] = [(q, w_ans)]
            else:
                self.workers[w].append((q, w_ans))
            #end if

            # add worker's answer to a general unmanaged list of all answers
            self.answers.append((q, w, w_ans))
        #end for
    #end constructor
#end class

