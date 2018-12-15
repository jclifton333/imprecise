"""
Resolve finite imprecise credence in two-action setting via optimization over stochastic matrices.
"""

import numpy as np
from gurobipy import *


def get_best_matrix(P, u_1, u_2):
  """
  Get stochastic matrix closest to P such that u_1 > u_2 or u_2 > u_1; involves two quadratic programs.

  :param P: nxm matrix, where n=num probability distributions and m=number of models.
  :param u_1: vector of utilities under each model given action 1
  :param u_2: "" "" given action 2
  :return:
  """

  n, m = P.shape

  # Model for a1
  model_1 = Model("a1")
  # model_1.setParam('OutputFlag', False)

  # Define decision variables
  vars_ = [[] for i in range(n)]
  for i in range(n):
    for j in range(m):
      vars_[i].append(model_1.addVar(lb=0.0, ub=1.0, vtype=GRB.SEMICONT))

  # Define objective
  obj = QuadExpr()
  for i in range(n):
    for j in range(m):
      obj += (vars_[i][j] - P[i, j])**2
  model_1.setObjective(obj)

  # Define sum-to-1 constraint
  sum_constr_expr = LinExpr()
  for i in range(n):
    sum_constr_expr.addTerms([1.0]*m, vars_[i])
    model_1.addConstr(sum_constr_expr == 1)

  # Define u(a_1) > u(a_2) constraint
  ineq_constr_expr = LinExpr()
  for i in range(n):
    ineq_constr_expr.addTerms(vars_[i], u_1 - u_2)
    model_1.addConstr(ineq_constr_expr >= 0)

  model_1.optimize()
  return np.array([np.array([var_ for var_ in  row] for row in vars_)])


if __name__ == "__main__":
  P_ = np.random.dirichlet(alpha=[0.1, 0.6, 0.3], size=10)
  u_1_ = np.array([1, 2, -1])
  u_2_ = np.array([0, -1, 3])
  get_best_matrix(P_, u_1_, u_2_)
