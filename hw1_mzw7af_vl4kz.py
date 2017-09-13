#!/usr/bin/python

from collections import OrderedDict
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
    print(varDef)


def teachRootVar(root_var, bool_val):
    '''
    Teach <ROOTVAR> = <BOOL>
    '''
    global facts
    if varDef[root_var][0] == "-L":
        raise Exception("Error, trying to set non-root variable")
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
    pass


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
    pass

if __name__ == "__main__":
    main()
