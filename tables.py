ADDITIVE = "ADDITIVE"
COMPLEX = "COMPLEX"
CONSTANT = "CONSTANT"
SMALL = "SMALL"

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

_SMALL_TABLE = [[0, 0, 0],
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 2]]

TABLES = {CONSTANT: _CONSTANT_TABLE,
          ADDITIVE: _ADDITIVE_TABLE,
          COMPLEX: _COMPLEX_TABLE,
          SMALL: _SMALL_TABLE
         }

def getTable(tableType):
    return TABLES[tableType]
