"""
Resolve finite imprecise credence in two-action setting via optimization over stochastic matrices.

Given imprecise credence matrix P and utilities u_1, u_2, under actions 1, 2 resp the problem is

min distance(P', P)
s.t. P' stochastic
     P'(u_1 - u_2) > 0, or
     P'(u_2 - u_1) > 0.
"""

import numpy as np
import pdb
from gurobipy import *


def l1_matrix_opt_for_direction(P, u_1, u_2, direction):
  """
  Get stochastic matrix closest to P in l1 such that u_1 > u_2 or u_2 > u_1.

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
  pos_part = [[] for i in range(n)]
  neg_part = [[] for i in range(n)]
  vars_ = [[] for i in range(n)]
  for i in range(n):
    for j in range(m):
      pos_part[i].append(model_1.addVar(lb=0.0, ub=1.0, vtype='C'))
      neg_part[i].append(model_1.addVar(lb=0.0, ub=1.0, vtype='C'))
      vars_[i].append(model_1.addVar(lb=0.0, ub=1.0, vtype='C'))

  # Define objective
  obj = LinExpr()
  for i in range(n):
    for j in range(m):
      obj += pos_part[i][j] + neg_part[i][j] # Absolute value of difference between P and decision matrix
  model_1.setObjective(obj, GRB.MINIMIZE)

  # Define pos part + neg part = x sum-to-1 constraint
  for i in range(n):
    # sum_constr_expr = LinExpr()
    # sum_constr_expr.addTerms([1.0]*m, [-pos_part[i][j] + neg_part[i][j] for j in range(m)])
    # model_1.addConstr(sum_constr_expr == 1 - P[i, :].sum())
    model_1.addConstr(quicksum([-pos_part[i][j] + neg_part[i][j] for j in range(m)]) == 1 - P[i, :].sum())

  # Define u(a_1) > u(a_2) constraint
  for i in range(n):
    # ineq_constr_expr = LinExpr()
    # if direction == 1:
    #   ineq_constr_expr.addTerms(u_1 - u_2, [-pos_part[i][j] + neg_part[i][j] for j in range(m)])
    # elif direction == 2:
    #   ineq_constr_expr.addTerms(u_2 - u_1, [-pos_part[i][j] + neg_part[i][j] for j in range(m)])
    if direction == 1:
      model_1.addConstr(quicksum([(u_1[j] - u_2[j])*(-pos_part[i][j] + neg_part[i][j]) for j in range(m)]) >= -np.dot(u_1 - u_2, P[i, :]))
    elif direction == 2:
      model_1.addConstr(quicksum([(u_2[j] - u_1[j])*(-pos_part[i][j] + neg_part[i][j]) for j in range(m)]) >= -np.dot(u_2 - u_1, P[i, :]))

  # Define pos part, neg part > 0 constraint
  for i in range(n):
    for j in range(m):
      model_1.addConstr(P[i, j] - pos_part[i][j] + neg_part[i][j] >= 0)

  model_1.optimize()
  best_matrix = np.array([[P[i, j] - pos_part[i][j].X + neg_part[i][j].X  for j in range(m)] for i in range(n)])
  objective_ = model_1.getObjective().getValue()
  return {'best_matrix': best_matrix, 'objective': objective_}


def l2_matrix_opt_for_direction(P, u_1, u_2, direction):
  """
  Get stochastic matrix closest to P in l2 such that u_1 > u_2 or u_2 > u_1.

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
      vars_[i].append(model_1.addVar(lb=0.0, ub=1.0, vtype='C'))

  # Define objective
  obj = QuadExpr()
  for i in range(n):
    for j in range(m):
      obj += (vars_[i][j] - P[i, j]) * (vars_[i][j] - P[i, j])
  model_1.setObjective(obj, GRB.MINIMIZE)

  # Define sum-to-1 constraint
  for i in range(n):
    sum_constr_expr = LinExpr()
    sum_constr_expr.addTerms([1.0]*m, vars_[i])
    model_1.addConstr(sum_constr_expr == 1)

  # Define u(a_1) > u(a_2) constraint
  for i in range(n):
    ineq_constr_expr = LinExpr()
    if direction == 1:
      ineq_constr_expr.addTerms(u_1 - u_2, vars_[i])
    elif direction == 2:
      ineq_constr_expr.addTerms(u_2 - u_1, vars_[i])
    model_1.addConstr(ineq_constr_expr >= 0)

  model_1.optimize()
  best_matrix = np.array([[var_.X for var_ in row] for row in vars_])
  objective_ = model_1.getObjective().getValue()
  return {'best_matrix': best_matrix, 'objective': objective_}


def get_best_matrix(P, u_1, u_2, metric='l2'):
  if metric == 'l2':
    results_for_u1_gr_u2 = l2_matrix_opt_for_direction(P, u_1, u_2, 1)
    results_for_u2_gr_u1 = l2_matrix_opt_for_direction(P, u_1, u_2, 2)
  elif metric == 'l1':
    results_for_u1_gr_u2 = l1_matrix_opt_for_direction(P, u_1, u_2, 1)
    results_for_u2_gr_u1 = l1_matrix_opt_for_direction(P, u_1, u_2, 2)
  return results_for_u1_gr_u2, results_for_u2_gr_u1


if __name__ == "__main__":
  np.random.seed(3)
  P_ = np.random.dirichlet(alpha=[0.1, 0.6, 0.3], size=10)
  u_1_ = np.array([1, 2, -1])
  u_2_ = np.array([0, -1, 3])
  res_1, res_2 = get_best_matrix(P_, u_1_, u_2_, metric='l1')
