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
        if var not in varDef and var != "":
            return
    # if result var is unknown or is not a learned variable, skip
    if variable not in varDef or varDef[variable][0] != "-L":
        return
    rules.append((expr, variable))


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
    global facts

    while True:
        factAdded = False

        for index, rule in enumerate(rules):
            if evalExpr(rule[0]) and rule[1] not in facts.keys():
                factAdded = True
                facts[rule[1]] = index

        if not factAdded:
            break

def query(expr):
    '''
    Query <EXP>
    '''
    truth, _ = whyHelper(expr, set())
    if truth:
        print("true")
    else:
        print("false")


def why(expr):
    '''
    Why <EXP>
    '''
    truth, explanation = whyHelper(expr, set())
    if truth:
        print("true")
    else:
        print("false")
    print(explanation.strip())


def whyHelper(expr, ruleSet):
    '''
    Helper function for Why. Performs the same function, but also returns the text
    detailing why the expr can/cannot be proved.
    Also passes a set of rule indices that have already been used, for cycle detection
    Returns a tuple: (bool truth, string explanation)
    '''
    # expr is a variable. if it is a root var than we're good. Else
    # look for rule(s) that implies it's true
    if re.fullmatch('[a-zA-Z_]+', expr) is not None:
        if varDef[expr][0] == "-R":
            truth = expr in facts.keys()
            return (truth, printStatement('fact', truth, expr))
        else:
            # We know this var is true and we have link to rule that implies true
            if expr in facts.keys():
                possible_rules = [rules[facts[expr]]]
            # else get all rules that imply this is true
            else:
                possible_rules = [r for r in rules if r[1] == expr]
            # loop through rules and return the first true rule
            # or return the last false rule
            for index, rule in enumerate(possible_rules):
                if rule in ruleSet:
                    continue
                truth, text = whyHelper(rule[0], ruleSet | {rule})
                if truth or index == len(possible_rules) - 1:
                    text += printStatement('rule', truth, rule[0], rule[1])
                    return (truth, text)
            # No rules conclude this variable is true, so return it is false
            return (False, printStatement('fact', False, expr))
    # expr is an expression
    split, op = splitExpr(expr)
    if op == '&':
        truth1, exp1 = whyHelper(split[0], ruleSet)
        truth2, exp2 = whyHelper(split[1], ruleSet)
        if truth1 and truth2:
            text = exp1 + exp2 + printStatement('conclude', True, expr)
        # one of the sides is false. Show the first if both are false
        elif not truth1:
            text = exp1 + printStatement('conclude', False, expr)
        else:
            text = exp2 + printStatement('conclude', False, expr)
        return (truth1 and truth2, text)
    elif op == '|':
        truth1, exp1 = whyHelper(split[0], ruleSet)
        truth2, exp2 = whyHelper(split[1], ruleSet)
        if not (truth1 or truth2):
            text = exp1 + exp2 + printStatement('conclude', False, expr)
        # One of the sides is true. Show the first if both are true
        elif truth1:
            text = exp1 + printStatement('conclude', True, expr)
        else:
            text = exp2 + printStatement('conclude', True, expr)
        return (truth1 or truth2, text)
    elif op == '!':
        truth, exp = whyHelper(split[1], ruleSet)
        text = exp + printStatement('conclude', not truth, expr)
        return (not truth, text)


def printStatement(logicType, truth, expr1, expr2=None):
    '''
    Helper function to print 'Why' reasoning. Takes in 3 (or 4) arguments:
    1. logicType: a string indicating (fact, rule, conclude)
    2. truth value (true or false)
    3. expr1 that was proved true or false (or conditional for rule)
    4. expr2 (if rule, then the implied expr)
    Returns the resulting reasoning string
    '''
    expr1 = formatExprPrint(expr1)
    if logicType == 'fact':
        if truth:
            return "I KNOW THAT %s\n" % expr1
        else:
            return "I KNOW IT IS NOT TRUE THAT %s\n" % expr1
    if logicType == 'rule':
        expr2 = formatExprPrint(expr2)
        if truth:
            return "BECAUSE %s I KNOW THAT %s\n" % (expr1, expr2)
        else:
            return "BECAUSE IT IS NOT TRUE THAT %s I CANNOT PROVE %s\n" % (expr1, expr2)
    if logicType == 'conclude':
        if truth:
            return "I THUS KNOW THAT %s\n" % (expr1)
        else:
            return "THUS I CANNOT PROVE %s\n" % (expr1)


def formatExprPrint(expr):
    '''
    Format expr to print ready form
    '''
    def matchRepl(matchobj):
        return varDef[matchobj.group(0)][1]

    expr = re.sub('[a-zA-Z_]+', matchRepl, expr)
    expr = expr.replace('&', ' AND ')
    expr = expr.replace('|', ' OR ')
    expr = expr.replace('!', 'NOT ')
    return expr


def evalExpr(expr):
    '''
    Evaluate an expression taking into account order of operations & parentheses
    Order of operations: NOT, AND, OR
    '''

    if re.fullmatch('[a-zA-Z_]+', expr) is not None:
        if expr in facts.keys():
            return True
        else:
            return False

    split, operator = splitExpr(expr)

    if operator == "&":
        return evalExpr(split[0]) and evalExpr(split[1])
    elif operator == "!":
        return not evalExpr(split[1])
    elif operator == "|":
        return evalExpr(split[0]) or evalExpr(split[1])


def splitExpr(expr):
    '''
    Splits expressions based on order of operations
    Returns array of split expression and operator the array was split on
    '''
    opList = ["|", "&", "!"]

    if expr[0] == "(" and expr[len(expr)-1] == ")":
        expr = expr[1:len(expr)-1]

    for operator in opList:
        parenCount = 0
        for index, char in enumerate(expr):
            if char == "(":
                parenCount += 1
            if char == ")":
                parenCount -= 1

            if parenCount == 0 and char == operator:
                split = [expr[0:index], expr[(index+1):]]
                return (split, operator)


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
