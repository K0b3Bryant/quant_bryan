import numpy as np

def ep_optimal_portfolios(expected_returns, cov_matrix):
  from deap import base, creator, tools, algorithms

  n_assets = len(expected_returns)

  # Define fitness as maximizing Sharpe Ratio
  def sharpe_ratio(weights):
      weights = np.array(weights)
      port_return = np.dot(weights, expected_returns)
      port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
      return port_return / port_vol,
  
  # Setup DEAP
  creator.create("FitnessMax", base.Fitness, weights=(1.0,))
  creator.create("Individual", list, fitness=creator.FitnessMax)
  
  toolbox = base.Toolbox()
  toolbox.register("attr_float", np.random.dirichlet, np.ones(n_assets)) # Dirichlet distribution so they sum to 1
  toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
  toolbox.register("population", tools.initRepeat, list, toolbox.individual)
  toolbox.register("evaluate", sharpe_ratio)
  toolbox.register("mate", tools.cxBlend, alpha=0.5)
  toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2) # Gaussian noise to weights
  toolbox.register("select", tools.selTournament, tournsize=3) # Pick top 3 survivors

  # Run optimization
  pop = toolbox.population(n=100)
  result, _ = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=50, verbose=False) # 50 generations, 0.7 crossover chance (kind of convergence), 0.3 mutation chance (kind of learning rate)

  # Show best portfolio
  best_portfolio = tools.selBest(result, k=1)[0]
  print("Optimal weights:", np.round(best_portfolio, 3))

  return best_portfolio

if __name__ == "__main__":
  # Simulated returns and covariance matrix
  expected_returns = np.array([0.08, 0.12, 0.10, 0.09])
  cov_matrix = np.array([[0.10, 0.02, 0.01, 0.03],
                         [0.02, 0.12, 0.04, 0.02],
                         [0.01, 0.04, 0.09, 0.01],
                         [0.03, 0.02, 0.01, 0.11]])
                         
  best_portfolio = ep_optimal_portfolios(expected_returns, cov_matrix)
