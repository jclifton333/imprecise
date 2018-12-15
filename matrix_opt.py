"""
Resolve finite imprecise credence in two-action setting via optimization over stochastic matrices.

Given imprecise credence matrix P and utilities u_1, u_2, under actions 1, 2 resp the problem is

min distance(P', P)
s.t. P' stochastic
     P'(u_1 - u_2) > 0, or
     P'(u_2 - u_1) > 0.
"""

import numpy as np
from gurobipy import *


def get_best_matrix_for_direction(P, u_1, u_2, direction):
  """
  Get stochastic matrix closest to P such that u_1 > u_2 or u_2 > u_1.

  :param P: nxm matrix, where n=num probability distributions and m=number of models.
  :param u_1: vector of utilities under each model given action 1
  :param u_2: "" "" given action 2
  :param direction: integer in [1, 2] corresponding to whether to constrain to u_1 > u_2 or u_2 > u_1
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
      obj += (vars_[i][j] - P[i, j]) * (vars_[i][j] - P[i, j])
  model_1.setObjective(obj)

  # Define sum-to-1 constraint
  sum_constr_expr = LinExpr()
  for i in range(n):
    sum_constr_expr.addTerms([1.0]*m, vars_[i])
    model_1.addConstr(sum_constr_expr == 1)

  # Define u(a_1) > u(a_2) constraint
  ineq_constr_expr = LinExpr()
  for i in range(n):
    if direction == 1:
      ineq_constr_expr.addTerms(vars_[i], u_1 - u_2)
    elif direction == 2:
      ineq_constr_expr.addTerms(vars_[i], u_2 - u_1)
    model_1.addConstr(ineq_constr_expr >= 0)

  model_1.optimize()
  best_matrix = np.array([np.array([var_ for var_ in row] for row in vars_)])
  objective_ = model_1.getObjective().getValue()
  return {'best_matrix': best_matrix, 'objective': objective_}


def get_best_matrix(P, u_1, u_2):
  results_for_u1_gr_u2 = get_best_matrix_for_direction(P, u_1, u_2, 1)
  results_for_u2_gr_u1 = get_best_matrix_for_direction(P, u_1, u_2, 2)
  return results_for_u1_gr_u2, results_for_u2_gr_u1


if __name__ == "__main__":
  P_ = np.random.dirichlet(alpha=[0.1, 0.6, 0.3], size=10)
  u_1_ = np.array([1, 2, -1])
  u_2_ = np.array([0, -1, 3])
  res_1, res_2 = get_best_matrix(P_, u_1_, u_2_)
