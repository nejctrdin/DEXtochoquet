import tables
import choquetOptimize
import sys


def printHelp(exitStatus, additionalText=""):
    if additionalText:
        print additionalText
        print
    print "The program is called as: python run.py [-h] optimization table"
    print "  -h:\t\t Print help"
    print "  optimization:\t Selected optimization\n\t\t LP\n\t\t QCP\n\t\t NLP"
    print "  table:\t Selected table\n\t\t ADDITIVE\n\t\t CONSTANT\n\t\t COMPLEX\n\t\t SMALL"
    sys.exit(exitStatus)

def main(args):
    if "-h" in args:
        printHelp(0)

    if 3 != len(args):
        printHelp(1, "Incorrect number of arguments")
    
    optimization, tableType = args[1:3]
    if tableType not in tables.TABLES:
        printHelp(2, 
                  ("Selected table type ({0}) is not available! "
                   "Possible options are {1}.").format(tableType,
                                                       sorted(list(tables.TABLES))))

    if optimization not in choquetOptimize.OPTIMIZATIONS:
        printHelp(2, 
                  ("Selected optimization type ({0}) is not available! " 
                   "Possible options are {1}.").format(optimization,
                                                      sorted(list(choquetOptimize.OPTIMIZATIONS))))

    decisionTable = tables.getTable(tableType)
    statusCode, status, computed = [0] * 3

    statusCode, status, computed = choquetOptimize.getChoquetCapacities(decisionTable, optimization)
    
    print "The problem is: {0}".format(status)
    print 
    print "Computed values are:"
    for name in sorted(list(computed)):
        print "{0} = {1}".format(name, computed[name])

if __name__ == "__main__":
    main(sys.argv)
