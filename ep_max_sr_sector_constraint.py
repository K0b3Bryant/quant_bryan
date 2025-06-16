import numpy as np

def ep_optimal_portfolios(expected_returns, cov_matrix, sector_labels, sector_max_exposure):
    from deap import base, creator, tools, algorithms

    n_assets = len(expected_returns)

    def sharpe_ratio_with_sector_penalty(weights):
        weights = np.array(weights)
        weights = np.clip(weights, 0, 1)
        if np.sum(weights) == 0:
            return -999.0,

        weights /= np.sum(weights)

        port_return = np.dot(weights, expected_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = port_return / port_vol if port_vol > 0 else -999.0

        # --- Sector exposure penalty ---
        sector_weights = {}
        for i, w in enumerate(weights):
            sector = sector_labels[i]
            sector_weights[sector] = sector_weights.get(sector, 0) + w

        penalty = 0.0
        for sector, total_weight in sector_weights.items():
            max_allowed = sector_max_exposure.get(sector, 1.0)
            if total_weight > max_allowed:
                penalty += (total_weight - max_allowed) * 10

        return sharpe - penalty,

    # Setup DEAP
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.dirichlet, np.ones(n_assets))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", sharpe_ratio_with_sector_penalty)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=100)
    result, _ = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=50, verbose=False)

    best_portfolio = tools.selBest(result, k=1)[0]
    print("Optimal weights:", np.round(best_portfolio, 3))

    return best_portfolio


if __name__ == "__main__":
    expected_returns = np.array([0.08, 0.12, 0.10, 0.09])
    cov_matrix = np.array([[0.10, 0.02, 0.01, 0.03],
                           [0.02, 0.12, 0.04, 0.02],
                           [0.01, 0.04, 0.09, 0.01],
                           [0.03, 0.02, 0.01, 0.11]])

    sector_labels = ['tech', 'tech', 'healthcare', 'finance']
    sector_max_exposure = {
        'tech': 0.4,
        'healthcare': 0.5,
        'finance': 0.5
    }

    best_portfolio = ep_optimal_portfolios(expected_returns, cov_matrix, sector_labels, sector_max_exposure)
