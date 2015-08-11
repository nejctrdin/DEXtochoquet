import unittest
import choquetOptimize

class TestChoquetOptimizations(unittest.TestCase):

    _NO_ROWS = "The table is not defined"
    _TABLE_EXPECTED = "The table must be of type list"
    _NOT_ENOUGH_ARGUMENTS = ("Table rows must have at least one argument "
                             "and one output")
    _NOT_EQUAL_ROWS = "Rows in table must have equal length"
    _UNKNOWN_PROGRAM = "Unknown program type specified"
    _QCP_ERROR = ("Problem during execution (CPLEX Error  5002: Q in 'row0' "
                  "is not positive semi-definite.)!")

    _ADDITIVE_TABLE = [[0, 0, 0],
                       [0, 1, 1],
                       [1, 0, 1],
                       [1, 1, 2]]

    def test_additive_LP(self):
        expectedValues = {"c00": 0.0, "c01": 0.001, "c11": 1.0, "c10": 0.001,
                          "d2": 1.0, "d0": 0.0, "d1": 0.001} 
        status, namedStatus, values = choquetOptimize.getChoquetCapacities(self._ADDITIVE_TABLE,
                                                                           "LP")

        self.assertEqual(status, 1)
        self.assertEqual(namedStatus, "Optimal")
        self.assertDictEqual(values, expectedValues)

    def test_errors(self):
        with self.assertRaisesRegexp(Exception, self._TABLE_EXPECTED):
            choquetOptimize.getChoquetCapacities({}, "LP")

        with self.assertRaisesRegexp(Exception, self._NO_ROWS):
            choquetOptimize.getChoquetCapacities([], "LP")

        with self.assertRaisesRegexp(Exception, self._NOT_ENOUGH_ARGUMENTS):
            choquetOptimize.getChoquetCapacities([[]], "LP")

        with self.assertRaisesRegexp(Exception, self._NOT_ENOUGH_ARGUMENTS):
            choquetOptimize.getChoquetCapacities([[1], [2]], "LP")

        with self.assertRaisesRegexp(Exception, self._NOT_EQUAL_ROWS):
            choquetOptimize.getChoquetCapacities([[1, 1], [2, 3, 4]], "LP")

        with self.assertRaisesRegexp(Exception, self._UNKNOWN_PROGRAM):
            choquetOptimize.getChoquetCapacities([[1, 1, 2], [2, 3, 4]], "P")

    def test_additive_QCP(self):
        status, namedStatus, values = choquetOptimize.getChoquetCapacities(self._ADDITIVE_TABLE,
                                                                           "QCP")
        self.assertEqual(status, -1)
        self.assertEqual(namedStatus, self._QCP_ERROR)
        
        for v in values:
            self.assertEqual(values[v], 0)

if __name__ == "__main__":
    unittest.main()
