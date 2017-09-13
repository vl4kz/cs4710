#!/usr/bin/python

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
varDef = {}
facts = {}
rules = []

# varDef = {
#     'S' : ("-R", "Sam likes Ice Cream"),
#     'V' : ("-R", "Today is Sunday"),
#     "EAT" : ("-L", "Sam will eat ice cream"),
# }
# facts = {
#     'S' : -1,
#     'V' : -1,
# }
# rules = [
#     ("S&V", "EAT"),
]

def teachVar(argument, variable, definition):
    '''
    Teach <ARG> <VAR> = <STRING>
    '''
    global varDef;
    varDef[variable] = (argument, definition);


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
