# -*- coding: utf-8 -*-

from .workspace import Workspace, get_label_continuous
import random
import numpy as np
import sys
# import spot

def convert_formula(formula):
    replacements = {
        '||': '|',
        '&&': '&',
        '[]': 'G',
        '<>': 'F',
        'NOT': '!',
        'TRUE': '1',
        'FALSE': '0'
    }
    
    for new, old in replacements.items():
        formula = formula.replace(old, new)
    
    new_str = ""
    for ch in formula:
        if 'a' <= ch <= 'z':
            new_str += 'e' + str(ord(ch) - 96)
        else:
            new_str += ch
    
    formula = new_str.replace('TRUE', 'true').replace('FALSE', 'false')
    
    return formula

def get_random_formula(type=0, load_path="./utils/LTL.txt"):
    # load formula from file
    if type == 0:
        with open(load_path, 'r') as f:
            formulas = f.readlines()
            formula = random.choice(formulas).strip()
    
    return convert_formula(formula)




class Task(object):
    """
    define the task specified in LTL
    """
    def __init__(self, workspace, LTL):
        """
        +----------------------------+
        |   Propositonal Symbols:    |
        |       true, false         |
        |	    any lowercase string |
        |                            |
        |   Boolean operators:       |
        |       !   (negation)       |
        |       ->  (implication)    |
        |       &&  (and)            |
        |       ||  (or)             |
        |                            |
        |   Temporal operators:      |
        |       []  (always)         |
        |       <>  (eventually)     |
        |       U   (until)          |
        +----------------------------+
        """
        self.formula = LTL
        self.subformula = {1: '(l1_1)',
                           2: '(l2_1)',
                           3: '(l3_1)',
                           4: '(l4_1)',
                           5: '(l5_1)',
                           6: '(l6_1)',
                           7: '(l7_1)',
                           }
        self.number_of_robots = 1 # only consider one robot in this project

        self.init = []  # initial locations
        self.init_label = []  # labels of initial locations
        while True:
            ini = (random.uniform(0, workspace.continuous_workspace[0]), random.uniform(0, workspace.continuous_workspace[1]))
            ap = get_label_continuous(ini, workspace)
            if 'o' not in ap:
                break
        self.init = tuple(ini)
        workspace.init = self.init
        ap = ap + '_' + str(1) if 'l' in ap else ''
        self.init_label.append(ap)
        self.useful = []  # useful subformulas that appear in the formula
        for i in range(7):
            if self.formula.find('e' + str(i + 1)) != -1:
                self.useful.append('l{}'.format(i+1))
