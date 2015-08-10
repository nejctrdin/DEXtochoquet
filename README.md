# DEX to Discrete Choquet capacities

This is a python utility, which takes a [*DEX*](http://kt.ijs.si/MarkoBohanec/dexi.html) function (a collection of discrete *if-then* rules) and tries to compute the capacities used for the discrete version of the [*choquet integral*](https://en.wikipedia.org/wiki/Choquet_integral). It does this using *linear programming* or *quadratic constraint programming*, specified by the arguments to the function.

## Prerequisites

For running the optimization problems [cplex](http://www-01.ibm.com/software/commerce/optimization/cplex-optimizer/) (quadratic constraint programming), [pulp](https://pypi.python.org/pypi/PuLP) and [GLPK](http://www.gnu.org/software/glpk/) (linear programming) are needed.

## Usage

```bash
The program is called as: python run.py [-h] optimization table
  -h:           Print help
  optimization: Selected optimization
                LP
                QCP
  table:        Selected table
                ADDITIVE
                CONSTANT
                COMPLEX
```

## License
    Copyright (C) 2012-2015 Nejc Trdin

        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program.  If not, see <http://www.gnu.org/licenses/>.
