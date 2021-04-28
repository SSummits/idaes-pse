##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2019, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# Please see the files COPYRIGHT.txt and LICENSE.txt for full copyright and
# license information, respectively. Both files are also available online
# at the URL "https://github.com/IDAES/idaes-pse".
##############################################################################

'''
Consider the following optimization problem.

min f:  p1*x1+ p2*(x2^2) + p1*p2
         s.t  c1: x1 + x2 = p1
              c2: x2 + x3 = p2
              0 <= x1, x2, x3 <= 10
              p1 = 10
              p2 = 5

Variables = (x1, x2, x3)
Parameters (fixed variables) = (p1, p2)
'''

import pyomo.environ as pyo
import numpy as np
from idaes.apps.uncertainty_propagation.uncertainties import propagate_uncertainty

### Create optimization model
m = pyo.ConcreteModel()
m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

# Define variables
# m.x1 = pyo.Var(bounds=(0,10))
# m.x2 = pyo.Var(bounds=(0,10))
# m.x3 = pyo.Var(bounds=(0,10))

m.x1 = pyo.Var()
m.x2 = pyo.Var()
m.x3 = pyo.Var()



# Define parameters
# m.p1 = pyo.Param(initialize=10, mutable=True)
# m.p2 = pyo.Param(initialize=5, mutable=True)
m.p1 = pyo.Var(initialize=10)
m.p2 = pyo.Var(initialize=5)

# Define constraints
m.con1 = pyo.Constraint(expr=m.x1 + m.x2 == m.p1)
m.con2 = pyo.Constraint(expr=m.x2 + m.x3 == m.p2)

# Define objective
m.obj = pyo.Objective(expr=m.p1*m.x1+ m.p2*(m.x2**2) + m.p1*m.p2, sense=pyo.minimize)

### Solve optimization model
opt = pyo.SolverFactory('ipopt',tee=True)
opt.solve(m)


### Inspect solution
print("Numeric solution:")
print("x1 =",m.x1())
print("x2 =",m.x2())
print("x3 =",m.x3())
print(m.dual.display())

### Analytic solution
'''
At the optimal solution, none of the bounds are active. As long as the active set
does not change (i.e., none of the bounds become active), the
first order optimality conditions reduce to a simple linear system.
'''

# dual variables (multipliers)
v2_ = 0
v1_ = m.p1()

# primal variables
x2_ = (v1_ + v2_)/(2 * m.p2())
x1_ = m.p1() - x2_
x3_ = m.p2() - x2_

print("\nAnalytic solution:")
print("x1 =",x1_)
print("x2 =",x2_)
print("x3 =",x3_)
print("v1 =",v1_)
print("v2 =",v2_)
 
### Analytic sensitivity
'''
Using the analytic solution above, we can compute the sensitivies of x and v to
perturbations in p1 and p2.


The matrix dx_dp constains the sensitivities of x to perturbations in p
''' 

# Initialize sensitivity matrix Nx x Np
# Rows: variables x
# Columns: parameters p
dx_dp = np.zeros((3,2))

# dx2/dp1 = 1/(2 * p1)
dx_dp[1, 0] = 1/(2*m.p1())

# dx2/dp2 = (v1 + v2)/(2 * p2**2)
dx_dp[1,1] = (v1_ + v2_)/(2 * m.p2()**2)

# dx1/dp1 = 1 - dx2/dp1
dx_dp[0, 0] = 1 - dx_dp[1,0]

# dx1/dp2 = 0 - dx2/dp2
dx_dp[0, 1] = 0 - dx_dp[1,1]

# dx3/dp1 = 1 - dx2/dp1
dx_dp[2, 0] = 1 - dx_dp[1,0]

# dx3/dp2 = 0 - dx2/dp2
dx_dp[2, 1] = 0 - dx_dp[1,1]

print("\n\ndx/dp =\n",dx_dp)


'''
Similarly, we can compute the gradients df_dx, df_dp
and Jacobians dc_dx, dc_dp
'''

# Initialize 1 x 3 array to store (\partial f)/(\partial x)
# Elements: variables x
df_dx = np.zeros(3)

# df/dx1 = p1
df_dx[0] = m.p1()

# df/dx2 = p2
df_dx[1] = 2 * m.p2() * x2_

# df/dx3 = 0

print("\n\ndf/dx =\n",df_dx)

# Initialize 1 x 2 array to store (\partial f)/(\partial p)
# Elements: parameters p
df_dp = np.zeros(2)

# df/dxp1 = x1 + p2
df_dp[0] = x1_ + m.p2()

# df/dp2 = 2 * p2 * x2 + p1
df_dp[1] = 2 * m.p2() * x2_ + m.p1()

print("\n\ndf/dp =\n",df_dp)

# Initialize 2 x 3 array to store (\partial c)/(\partial x)
# Rows: constraints c
# Columns: variables x
dc_dx = np.zeros((2,3))

# dc1/dx1 = 1
dc_dx[0,0] = 1

# dc1/dx2 = 1
dc_dx[0,1] = 1

# dc2/dx2 = 1
dc_dx[1,1] = 1

# dc2/dx3 = 1
dc_dx[1,2] = 1

# Remaining entries are 0

print("\n\ndc/dx =\n",dc_dx)

# Initialize 2 x 2 array to store (\partial c)/(\partial x)
# Rows: constraints c
# Columns: variables x
dc_dp = np.zeros((2,2))

# dc1/dp1 = -1
dc_dp[0,0] = -1

# dc2/dp2 = -1
dc_dp[1,1] = -1

# Remaining entries are 0

print("\n\ndc/dp =\n",dc_dp)

### Uncertainty propagation
'''
Now lets test the uncertainty propagation package. We will assume p has covariance
sigma_p = [[2, 0], [0, 1]]
'''

## Prepare inputs

# Covariance matrix
sigma_p = np.array([[2, 0], [0, 1]])

# Nominal values for uncertain parameters
theta = {'p1':m.p1(), 'p2':m.p2()}

# Names of uncertain parameters
theta_names = ['p1','p2']

## Run package
results = propagate_uncertainty(m, theta, sigma_p, theta_names)
