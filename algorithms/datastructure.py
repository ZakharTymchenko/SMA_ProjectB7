#!/usb/bin/python
"""
Basic class that forms the dataset
"""
class Dataset:
    """
    Since python is untyped here's a small reference about what can one expect
    in the lists and what's frequently used here in general:
        QUESTION (or Q) - a tuple of (ID1, ID2) where IDs are product ids
        WORKER - a string which denotes a worker answering questions
        ANSWER - an int which can only be a 0 or 1
    """

    """
    questions is a dictionary here
        KEY is a QUESTION
        VALUE is an ANSWER
    """
    questions = dict()

    """
    workers is a dictionary where
        KEY is WORKER
        VALUE is a list of tuples (QUESTION, ANSWER)
    """
    workers = dict()

    """
    answers is a list of tuples (QUESTION, WORKER, ANSWER)
    """
    answers = []
#end class

def ArrangeData(q_train, q_valid, q_test, workers, answers):
    train = Dataset()
    valid = Dataset()
    test = Dataset()

    train.questions = q_train
    valid.questions = q_valid
    test.questions = q_test

    train.workers = dict(
        (w, [q for q in wq if q[1] in q_train]) for (w, wq) in workers.items()
    )
    valid.workers = dict(
        (w, [q for q in wq if q[1] in q_valid]) for (w, wq) in workers.items()
    )
    test.workers = dict(
        (w, [q for q in wq if q[1] in q_test]) for (w, wq) in workers.items()
    )

    train.answers = [
        ans for ans in answers if ans[0] in q_train
    ]
    valid.answers = [
        ans for ans in answers if ans[0] in q_valid
    ]
    test.answers = [
        ans for ans in answers if ans[0] in q_test
    ]

    return (train, valid, test)
