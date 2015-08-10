import cplex
import re
import sys

EPS = 0.001

def getChoquetCapacities(table, printProblem):
    # create problem
    N = len(table[0]) - 1
    variables = []

    prob = cplex.Cplex()
    prob.objective.set_sense(prob.objective.sense.minimize)

    # create variables
    for i in xrange(2 ** N):
        s = bin(i)[2:]
        s = "c{0}{1}".format("0" * (N - len(s)), s)
        variables.append(s)
        
    prob.variables.add(names=variables)
    prob.variables.set_lower_bounds([(var, 0) for var in variables])
    prob.variables.set_upper_bounds([(var, 1.0) for var in variables])

    lin_expr = []
    senses = []
    rhs = []
    
    # normalization
    lin_expr.append(cplex.SparsePair(ind=["c{0}".format("0" * N)], val=[1.0]))
    lin_expr.append(cplex.SparsePair(ind=["c{0}".format("1" * N)], val=[1.0]))
    senses.append("E") 
    senses.append("E") 
    rhs.append(0)
    rhs.append(1.0)

    # monotonicity
    for lower in variables:
        # for each variable specify the dominating variable
        # each zero may be 1 and must dominate the particular lower variable
        indices = [m.start() for m in re.finditer("0", lower)]
        for i in indices:
            higher = "{0}1{1}".format(lower[:i], lower[i+1:])
            lin_expr.append(cplex.SparsePair(ind=[lower, higher], val=[1.0, -1.0]))
            senses.append("L") 
            rhs.append(0)
             
    # create a dictionary of output vaariables
    mapping = {}
    variables = []
    column = 0

    for column in xrange(N + 1):
        current = {}
        index = 0
        for row in xrange(len(table)):
            el = table[row][column]
            if el not in current:
                tmp = ""
                if column == N:
                    tmp = "d{0}{1}".format(index, column)
                else:
                    tmp = "v{0}{1}".format(index, column)
                index += 1
                current[el] = tmp
                variables.append(tmp)

        mapping[column] = current
    prob.variables.add(names=variables)

    prob.variables.set_lower_bounds([(var, 0) for var in variables])
    prob.variables.set_upper_bounds([(var, 10) for var in variables])

    for colIndex in mapping:
        sort = sorted(list(mapping[colIndex]))
        criterion = mapping[colIndex]
        # specify that lower value means it is dominated by the higher value
        for i in xrange(len(sort) - 1):
            lower = criterion[sort[i]]
            higher = criterion[sort[i + 1]]
            #prob += lower + EPS <= higher
            lin_expr.append(cplex.SparsePair(ind=[lower, higher], val=[1.0, -1.0]))
            senses.append("L") 
            rhs.append(-EPS)

    decisions = mapping[N]

    # the objective is to minimize the most preferrable decision
    prob.objective.set_linear(decisions[max(decisions)], 1.0)
    
    prob.linear_constraints.add(lin_expr=lin_expr,
                                senses=senses,
                                rhs=rhs) 
    
    # integrating the decision table
    rowIndex = 0
    for row in table:
        rawCriteria, dec = row[:-1], row[-1]
        decision = decisions[dec]
        
        criteria = []

        for i in xrange(len(rawCriteria)):
            criterionMap = mapping[i]
            criteria.append(criterionMap[rawCriteria[i]])

        # find the permutation which creates an increasing order of values
        permutation = {}
        for i in xrange(len(criteria)):
            permutation[i] = i

        sort = sorted(rawCriteria)
        if rawCriteria != sort:
            # must permute
            index = 0
            for c in rawCriteria:
                permIndex = sort.index(c)
                permutation[index] =  permIndex
                index += 1
                sort[permIndex] = -1
        
        # create choquet integral
        ind1 = []
        ind2 = []
        val = []
        for i in xrange(0, N):
            subset = ["0"] * N
            
            # create the elements used for capacity
            for j in xrange(i, N):
                subset[permutation[j]] = "1"

            # find the corresponding capacity for the selected elements
            var = "c{0}".format("".join(subset))

            if i==0:
                # value for the i = -1 is by definition 0
                # var * criteria[permutation[i]]
                ind1.append(var)
                ind2.append(criteria[permutation[i]])
                val.append(1.0)
            else:
                # var * (criteria[permutation[i]] - criteria[permutation[i-1]])
                ind1.append(var)
                ind2.append(criteria[permutation[i]])
                val.append(1.0)
                ind1.append(var)
                ind2.append(criteria[permutation[i - 1]])
                val.append(-1.0)
        
        lConstraint = cplex.SparsePair(ind=[decision], val=[-1.0])
        qConstraint = cplex.SparseTriple(ind1=ind1, ind2=ind2, val=val)
        
        # add the expression to the problem
        prob.quadratic_constraints.add(lin_expr=lConstraint, quad_expr=qConstraint, sense="E", rhs=0, name="row{0}".format(rowIndex))
        rowIndex += 1

    prob.solve()

    status = prob.solution.get_status()
    humanStatus = prob.solution.status[prob.solution.get_status()]
    
    values = {}
    cols = prob.variables.get_names()
    for j in range(len(cols)):
        values[cols[j]] = prob.solution.get_values(j)

    if printProblem:
        print prob

    return status, humanStatus, values
