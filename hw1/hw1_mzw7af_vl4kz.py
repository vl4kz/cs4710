#!/usr/bin/python

import sys
from collections import OrderedDict
import re

'''
GLOBAL VARIABLE DATA STRUCTURES
varDef - dict of variable definitions:
    {string var : (string arg, string definition) ]}
facts - dict of variables that are true and the index of the rule that led
    to proved it was true:
    {string var : int index, ...}
rules - list of rules:
    [(string expr, string var), ...]
'''
varDef = OrderedDict()
facts = OrderedDict()
rules = []


def teachVar(argument, variable, definition):
    '''
    Teach <ARG> <VAR> = <STRING>
    '''
    global varDef
    varDef[variable] = (argument, definition)


def teachRootVar(root_var, bool_val):
    '''
    Teach <ROOTVAR> = <BOOL>
    '''
    global facts
    if varDef[root_var][0] == "-L":
        print("Error, trying to set non-root variable")
        return
    if bool_val == 'true':
        facts[root_var] = -1
    else:
        facts.pop(root_var, None)

    # set all learned vars to false:
    for k, v in varDef.items():
        if v[0] == "-L":
            facts.pop(k, None)


def teachRule(expr, variable):
    '''
    Teach <EXP> -> <VAR>
    '''
    global rules
    expr_vars = re.split('[^a-zA-Z_]+', expr)

    # if variable in expression is unknown, skip this instr
    for var in expr_vars:
        if var not in varDef:
            print(var)
            return
    # if result var is unknown or is not a learned variable, skip
    if variable not in varDef or varDef[var][0] != "-L":
        return
    rules.append(expr, variable)


def listInst():
    '''
    List
    '''
    # list Root and list variables:
    print("Root Variables:")
    for k, v in varDef.items():
        if v[0] == "-R":
            print("\t%s = \"%s\"" % (k, v[1]))
    print()

    # list learned variables:
    print("Learned Variables:")
    for k, v in varDef.items():
        if v[0] == "-L":
            print("\t%s = \"%s\"" % (k, v[1]))
    print()

    # list facts:
    print("Facts:")
    for k in facts:
        print("\t%s" % (k))
    print()

    # list rules:
    print("Rules:")
    for rule in rules:
        print("\t%s -> %s" % (rule[0], rule[1]))
    print()


def learn():
    '''
    Learn
    '''
    pass


def query(expr):
    '''
    Query <EXP>
    '''
    pass


def why(expr):
    '''
    Why <EXP>
    '''
    pass


def main():
    for line in sys.stdin:
        argArray = line.strip().split(" ")
        command = argArray[0].lower()

        if command == "teach":
            if argArray[3] == "=":
                ruleDef = line.strip().split(" = ")[1]
                teachVar(argArray[1], argArray[2], ruleDef.replace('"', ""))
            elif argArray[2] == "=":
                teachRootVar(argArray[1], argArray[3])
            else:
                expression = line.strip().split(" ", 1)
                expVars = expression[1].split(" -> ")
                teachRule(expVars[0], expVars[1])
        elif command == "list":
            listInst()
        elif command == "learn":
            learn()
        elif command == "query":
            query(argArray[1])
        elif command == "why":
            why(argArray[1])

if __name__ == "__main__":
    main()
