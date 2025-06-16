import pandas as pd
from utils import validate_inputs, test_multiple_series
from config import CONFIG

def main():
    # Example inputs: Replace these with real data or dynamic input handling
    time_series_data = pd.DataFrame({
        'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
        'Series1': [100, 102, 101, 103, 104],
        'Series2': [200, 202, 201, 203, 204]
    })

    try:
        validated_data = validate_inputs(time_series_data)

        # Perform stationarity test for multiple series
        print("\nStationarity Test Results for Multiple Series:")
        multi_series_results = test_multiple_series(validated_data)
        for series_name, results in multi_series_results.items():
            print(f"\nResults for {series_name}:")
            for test_name, test_results in results.items():
                print(f"  {test_name}:")
                for key, value in test_results.items():
                    print(f"    {key}: {value}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
