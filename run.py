import choquetLP
import choquetQP
import sys

ADDITIVE = "ADDITIVE"
COMPLEX = "COMPLEX"
CONSTANT = "CONSTANT"

_ADDITIVE_TABLE = [[0, 0, 0],
                   [0, 1, 1],
                   [0, 2, 2],
                   [1, 0, 1],
                   [1, 1, 2],
                   [1, 2, 3],
                   [2, 0, 2],
                   [2, 1, 3],
                   [2, 2, 4]]

_COMPLEX_TABLE = [[0, 0, 0],
                  [0, 1, 1],
                  [0, 2, 1.5],
                  [1, 0, 1],
                  [1, 1, 2.5],
                  [1, 2, 3.5],
                  [2, 0, 1.5],
                  [2, 1, 2.5],
                  [2, 2, 4]]

_CONSTANT_TABLE = [[0, 0, 1],
                   [0, 1, 1],
                   [0, 2, 1],
                   [1, 0, 1],
                   [1, 1, 1],
                   [1, 2, 1],
                   [2, 0, 1],
                   [2, 1, 1],
                   [2, 2, 1]]

TABLES = {CONSTANT: _CONSTANT_TABLE,
          ADDITIVE: _ADDITIVE_TABLE,
          COMPLEX: _COMPLEX_TABLE
         }

_LP = "LP"
_QP = "QP"

OPTIMIZATIONS = {_LP, _QP}

def getTable(tableType):
    return TABLES[tableType]

def printHelp(exitStatus, additionalText=""):
    if additionalText:
        print additionalText
        print
    print "The program is called as: python run.py [-h] optimization table"
    print "  -h:\t\t Print help"
    print "  optimization:\t Selected optimization\n\t\t LP\n\t\t QCP"
    print "  table:\t Selected table\n\t\t ADDITIVE\n\t\t CONSTANT\n\t\t COMPLEX"
    sys.exit(exitStatus)

def main(args):
    if "-h" in args:
        printHelp(0)

    if 3 != len(args):
        printHelp(1, "Incorrect number of arguments")
    
    optimization, tableType = args[1:3]
    if tableType not in TABLES:
        printHelp(2, 
                  ("Selected table type ({0}) is not available! "
                   "Possible options are {1}.").format(tableType,
                                                       sorted(list(TABLES))))

    if optimization not in OPTIMIZATIONS:
        printHelp(2, 
                  ("Selected optimization type ({0}) is not available! " 
                   "Possible options are {1}.").format(optimization,
                                                      sorted(list(OPTIMIZATIONS))))

    decisionTable = getTable(tableType)
    statusCode, status, computed = [0] * 3

    if optimization == "LP":
        statusCode, status, computed = choquetLP.getChoquetCapacities(decisionTable, False)
    else:
        statusCode, status, computed = choquetQP.getChoquetCapacities(decisionTable, False)
    
    print "The problem is: {0}".format(status)
    print 
    print "Computed values are:"
    for name in sorted(list(computed)):
        print "{0} = {1}".format(name, computed[name])

if __name__ == "__main__":
    main(sys.argv)
