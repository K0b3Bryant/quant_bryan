import pandas as pd
from utils import validate_inputs, perform_factor_analysis, perform_nonlinear_analysis
from config import CONFIG

def main():
    # Example inputs: Replace these with real data or dynamic input handling
    returns_data = pd.DataFrame({
        'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'Returns': [0.02, 0.01, -0.01]
    })
    
    factors_data = pd.DataFrame({
        'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'Factor1': [0.1, 0.2, 0.15],
        'Factor2': [0.05, 0.07, 0.06]
    })

    try:
        validated_returns, validated_factors = validate_inputs(returns_data, factors_data)

        # Linear Factor Analysis
        print("\nLinear Factor Analysis Results:")
        linear_results = perform_factor_analysis(validated_returns, validated_factors)
        print(linear_results)

        # Nonlinear Factor Analysis
        print("\nNonlinear Factor Analysis Results:")
        nonlinear_results = perform_nonlinear_analysis(validated_returns, validated_factors)
        print(nonlinear_results)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
