"""
Resolve a collection of credences with a frequentist approach.
"""
import numpy as np
import scipy.optimize as optim
from functools import partial


def mse_optimal_weighting(alpha=np.array([1.1, 2, 3]), utility=np.array([1, 1, 1])):
  """
  Re-weight credence distribution to optimize utility estimation.

  :param alpha: Parameters of the dirichlet dbn we use to characterize the elicitor's credences over models.
  :param utility: Utilities associated to each model.
  :return:
  """
  number_of_models = len(alpha)
  U = np.diag(utility)
  alpha_0 = np.sum(alpha)
  p_bar = alpha / alpha_0
  covariance_matrix = np.zeros((number_of_models, number_of_models))
  denominator = alpha_0**3 + alpha_0**2
  for i in range(number_of_models):
    alpha_i = alpha[i]
    for j in range(i, number_of_models):
      if i == j:
        covariance_matrix[i, j] = (alpha_i*alpha_0 - alpha_i**2) / denominator
      else:
        alpha_j = alpha[j]
        covariance_matrix[i, j] = covariance_matrix[j, i] = (-alpha_i*alpha_j) / denominator

  def mse_objective(w, correct_model_index):
    """
    U = diag(u);
    p ~ Dirichlet(alpha);
    MSE = V( <w | U.p>) + E[( <w.U| p> - u[correct_model_index] )]^2
        = <w | U> . V(p) . <w | U> + (<w.U | E[p])^2 - 2*<w.U | E[p]>*u[correct_model_index]

    :param w:
    :return:
    """
    U_dot_w = np.dot(U, w)
    U_dot_w_dot_pbar = np.dot(U_dot_w, p_bar)
    variance = np.dot(U_dot_w.T, np.dot(covariance_matrix, U_dot_w))
    bias_squared = U_dot_w_dot_pbar**2 - 2*U_dot_w_dot_pbar*utility[correct_model_index]
    return variance + bias_squared

  mse_objective_part = partial(mse_objective, correct_model_index=2)
  res = optim.minimize(mse_objective_part, x0=np.random.random(size=number_of_models), bounds=[(0, None)
                                                                                       for _ in range(number_of_models)])
  return res


if __name__ == "__main__":
  res_ = mse_optimal_weighting()