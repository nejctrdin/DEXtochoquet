from pulp import *
import re

EPS = 0.001

def getChoquetCapacities(table, printProblem):
    # create problem
    N = len(table[0]) - 1
    variables = {}
    prob = LpProblem("choquet", LpMinimize)

    # create variables
    for i in xrange(2 ** N):
        s = bin(i)[2:]
        s = "0" * (N - len(s)) + s
        v = LpVariable("c{0}".format(s), 0)
        variables[s] = v
    
    # normalization
    prob += variables["0" * N] == 0
    prob += variables["1" * N] == 1

    # monotonicity
    for s in variables:
        # for each variable specify the dominating variable
        lower = variables[s]
        # each zero may be 1 and must dominate the particular lower variable
        indices = [m.start() for m in re.finditer("0", s)]
        for i in indices:
            tmp = "{0}1{1}".format(s[:i], s[i+1:])
            higher = variables[tmp]
            prob += lower <= higher

    # create mappings from criterion values, to [0, 1] interval
    criterionMapping = []

    for i in xrange(N):
        values = set()
        for row in table:
            values.add(row[i])

        values = sorted(list(values))
        minVal = values[0]
        maxVal = values[-1]
        normalizedValues = [float(val - minVal) / maxVal for val in values]
        mapping = {}
        for j in xrange(len(values)):
            mapping[values[j]] = normalizedValues[j]

        criterionMapping.append(mapping)

    # create a dictionary of output vaariables
    decisions = {}

    for row in table:
        # last item in the row is the decision
        decision = row[-1]
        if decision in decisions:
            continue
        var = LpVariable("d{0}".format(decision))
        decisions[decision] = var

    sort = sorted(list(decisions))

    # specify that lower value of the decision means it is dominated by the
    # higher valued decision
    for i in xrange(len(sort) - 1):
        lower = decisions[sort[i]]
        higher = decisions[sort[i + 1]]
        prob += lower + EPS <= higher

    # the objective is to minimize the most preferrable decision
    prob += decisions[sort[-1]]

    # integrating the decision table
    for row in table:
        rawCriteria, dec = row[:-1], row[-1]
        decision = decisions[dec]
        
        criteria = []

        for i in xrange(len(rawCriteria)):
            mapping = criterionMapping[i]
            criteria.append(mapping[rawCriteria[i]])

        # find the permutation which creates an increasing order of values
        permutation = {}
        for i in xrange(len(criteria)):
            permutation[i] = i

        sort = sorted(criteria)
        if criteria != sort:
            # must permute
            index = 0
            for c in criteria:
                permIndex = sort.index(c)
                permutation[index] =  permIndex
                index += 1
                sort[permIndex] = -1

        # each row in the table corresponds to one computation of choquet integral
        expression = 0

        # create choquet integral
        for i in xrange(0, N):
            subset = ["0"] * N
            
            # create the elements used for capacity
            for j in xrange(i, N):
                subset[permutation[j]] = "1"

            # find the corresponding capacity for the selected elements
            var = variables["".join(subset)]

            if i==0:
                # value for the i = -1 is by definition 0
                expression += var * criteria[permutation[0]]
            else:
                expression += var * (criteria[permutation[i]] - criteria[permutation[i-1]])

        # add the expression to the problem
        prob += expression == decision
   
    if printProblem:
        print prob

    # solving
    status = GLPK().solve(prob)

    # retrieving the results
    values = {}

    for v in prob.variables():
        values[v.name] =  v.varValue

    return status, LpStatus[status], values


