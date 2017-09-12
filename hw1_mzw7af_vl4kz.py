#!/usr/bin/python


def teachVar(argument, variable, definition):
    '''
    Teach <ARG> <VAR> = <STRING>
    '''
    pass

def teachRootVar(root_var, bool_val):
    '''
    Teach <ROOTVAR> = <BOOL>
    '''
    pass

def teachRule(expr, variable):
    '''
    Teach <EXP> -> <VAR>
    '''
    pass

def listInst():
    '''
    List
    '''
    pass

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
    '''
    varDef - dict of variable definitions:
        {string var : (string arg, string definition) ]}
    facts - dict of variables that are true and the index of the rule that led
        to proved it was true:
        {string var : int index, ...}
    rules - list of rules:
        [(string expr, string var), ...]
    '''
    varDef = {}
    facts = {}
    rules = []


if __name__ == "__main__":
    main()
