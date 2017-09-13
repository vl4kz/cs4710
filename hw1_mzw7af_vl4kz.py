#!/usr/bin/python
import sys

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

    for line in sys.stdin:
        argArray = line.strip().split(" ")

        if argArray[0].lower() == "teach":
            if argArray[3] == "=":
                stringVar = ' '.join(argArray).split(" = ")[1]
                teachVar(argArray[1], argArray[2], stringVar)
                #print(argArray[1] + " " + argArray[2] + " " + stringVar)
            elif argArray[2] == "=":
                teachRootVar(argArray[1], argArray[len(argArray)-1])
                #print(argArray[1] + " " + argArray[len(argArray)-1])
            else:
                expression = ' '.join(argArray[1:]).split(" -> ")
                teachRule(expression[0], expression[len(expression)-1])
                #print(expression[0] + " " + expression[len(expression)-1])
        elif argArray[0].lower() == "list":
            listInst()
        elif argArray[0].lower() == "learn":
            learn()
        elif argArray[0].lower() == "query":
            query(argArray[len(argArray)-1])
            #print(argArray[len(argArray)-1])
        elif argArray[0].lower() == "why":
            why(argArray[len(argArray)-1])
            #print(argArray[len(argArray)-1])
if __name__ == "__main__":
    main()
