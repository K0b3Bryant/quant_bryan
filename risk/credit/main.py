import pandas as pd
from utils import validate_inputs, merton_model, hull_white_model, ml_model, price_risky_bond, price_cds
from config import CONFIG

def main():
    # Example input data: Replace with actual data or dynamic input handling
    credit_data = pd.DataFrame({
        'Customer_ID': [1, 2, 3, 4],
        'Equity_Value': [50000, 60000, 45000, 70000],
        'Debt_Value': [20000, 15000, 30000, 10000],
        'Volatility': [0.2, 0.25, 0.3, 0.15],
        'Risk_Free_Rate': [0.03, 0.03, 0.03, 0.03],
        'Feature1': [0.5, 0.7, 0.6, 0.8],
        'Feature2': [0.2, 0.3, 0.4, 0.1]
    })

    try:
        validated_data = validate_inputs(credit_data)

        # Calculate Probability of Default using the selected model
        if CONFIG['default_probability_model'] == 'Merton':
            credit_data['PD'] = merton_model(validated_data)
        elif CONFIG['default_probability_model'] == 'Hull-White':
            credit_data['PD'] = hull_white_model(validated_data)
        elif CONFIG['default_probability_model'] == 'ML':
            credit_data['PD'] = ml_model(validated_data)
        else:
            raise ValueError("Invalid default probability model specified in CONFIG.")

        # Price a risky bond
        bond_price = price_risky_bond(
            face_value=1000,
            coupon_rate=0.05,
            maturity=5,
            risk_free_rate=0.03,
            pd=credit_data['PD'].mean(),
            recovery_rate=0.4
        )
        print(f"\nPrice of the risky bond: {bond_price:.2f}")

        # Price a credit default swap (CDS)
        cds_spread = price_cds(
            notional=1000,
            pd=credit_data['PD'].mean(),
            recovery_rate=0.4,
            risk_free_rate=0.03,
            maturity=5
        )
        print(f"\nCDS Spread: {cds_spread:.2f} basis points")

        print("\nCredit Risk Metrics:")
        print(credit_data)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
