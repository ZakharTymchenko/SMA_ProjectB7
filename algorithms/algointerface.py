#!/usr/bin/python
from datastructure import Dataset
"""
Inherit with:
    class MyAlgorithm(CrowdAlgorithm):
        ...

"""
class CrowdAlgorithm:
    # algorithm name
    name = ""

    train = Dataset()
    validation = Dataset()

    """
    This constructor is called from the subclass with:
        CrowdAlgorithm.__init__(self, "MyName", ...)
    Then all fields that subclass might need in addition are init'd
    No training is performed in the constructor.
    """
    def __init__(self, name, train, validation):
        self.name = name
        self.train = train
        self.validation = validation
    #end constructor


    """
    This function is overriden in the subclass with the same signature
    It should perform training using train/validation dicts
    """
    def fit(self):
        print("called abstract function train/0 in CrowdAlgorithm")
    #end function


    """
    This function is overriden in the subclass with the same signature
    ArgIn:
        -- testset : dict with KEY=QUESTION, VALUE=None
    """
    def test(self, testset):
        print("called abstract function test/1 in CrowdAlgorithm")
        return dict()
    #end function