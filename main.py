import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import numpy as np

all_data = []


def load_and_process_data(file_paths):
    global all_data
    all_data = []  # Clear in case the function is called multiple times

    for file_path in file_paths:
        data = pd.read_excel(file_path)

        # Filter data for the 'Güz' term
        filtered_data = data[data['DÖNEM'] == 'Güz'].copy()

        # Process the 'DERSLİK' column to extract floor information
        filtered_data['FLOOR'] = (
            filtered_data['DERSLİK']
            .str.extract(r'M (\d{1,3})')  # Extract the floor number
            .dropna()
            .astype(int) // 100  # Assuming the floor is in the first part of the room number
        )

        # Convert 'BAŞLANGIÇ SAATİ' to proper time format
        filtered_data['TIMESLOT'] = pd.to_datetime(
            filtered_data['BAŞLANGIÇ SAATİ'], format='%H:%M:%S', errors='coerce'
        ).dt.strftime("%H:%M")

        # Filter the weekdays
        filtered_data['DAY'] = filtered_data['GÜN']
        filtered_data = filtered_data[filtered_data['GÜN'].isin(['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma'])]

        # Rename columns for clarity
        filtered_data = filtered_data.rename(columns={
            'YIL': 'Year',
            'DERSI ALAN ÖĞRENCİ SAYISI': 'Number_of_People'
        })

        # Keep the necessary columns
        processed_data = filtered_data[['Year', 'DAY', 'TIMESLOT', 'FLOOR', 'DERSLİK', 'Number_of_People']]

        # Drop duplicates
        processed_data = processed_data.drop_duplicates()

        all_data.append(processed_data)

    # Combine all data from different files into one DataFrame
    combined_data = pd.concat(all_data, ignore_index=True).drop_duplicates()
    return combined_data


def train_and_evaluate_models(data):
    # Encode timeslots as numerical values
    data['TIMESLOT_NUM'] = data['TIMESLOT'].apply(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))

    X = data[['Year', 'TIMESLOT_NUM', 'FLOOR']]
    y = data['Number_of_People']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "KNN": KNeighborsRegressor(n_neighbors=3),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Support Vector Regression": SVR()
    }

    results = {}
    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mse = np.mean((y_test - y_pred) ** 2)
        results[model_name] = mse
        print(f"{model_name} - Mean Squared Error: {mse:.2f}")

    return models, scaler


# Predict 2024 values using different models and compare with actual values
def predict_2024(models, scaler, data_2024):
    data_2024['TIMESLOT_NUM'] = data_2024['TIMESLOT'].apply(
        lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1])
    )

    X_2024 = data_2024[['Year', 'TIMESLOT_NUM', 'FLOOR']]
    y_actual = data_2024['Number_of_People']

    X_2024_scaled = scaler.transform(X_2024)

    results_2024 = {}
    for model_name, model in models.items():
        y_pred = model.predict(X_2024_scaled)
        data_2024[f'Predicted_{model_name}'] = y_pred
        data_2024[f'Difference_{model_name}'] = y_pred - y_actual
        mse = np.mean((y_actual - y_pred) ** 2)
        results_2024[model_name] = mse

    return data_2024, results_2024


def save_predictions_to_excel(data_2024, results_2024, file_name="predictions_vs_models.xlsx"):
    with pd.ExcelWriter(file_name) as writer:
        data_2024.to_excel(writer, sheet_name="Predictions_vs_Actual", index=False)
        mse_df = pd.DataFrame(list(results_2024.items()), columns=["Model", "MSE"])
        mse_df.to_excel(writer, sheet_name="MSE_Results", index=False)
    print(f"Predictions and results saved to {file_name}")


def predict_2025_rf(data, model, scaler):
    """
    Predict 2025 values using Random Forest.
    """
    # Filter data for 2024 and prepare it for 2025
    data_2025 = data[data['Year'] == 2024].copy()
    data_2025['Year'] = 2025

    # Convert TIMESLOT to numeric format
    data_2025['TIMESLOT_NUM'] = data_2025['TIMESLOT'].apply(
        lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1])
    )

    # Prepare features
    X_2025 = data_2025[['Year', 'TIMESLOT_NUM', 'FLOOR']]
    X_2025_scaled = scaler.transform(X_2025)

    # Predict using Random Forest
    data_2025['Predicted_Random_Forest'] = model.predict(X_2025_scaled)

    # Drop actual values for clarity
    data_2025 = data_2025.drop(columns=['Number_of_People'], errors='ignore')

    return data_2025


def simulate_stair_usage(data):
    if 'DERSLİK' not in data.columns:
        raise ValueError("Column 'DERSLİK' is missing from the data.")

    data['ROOM_END'] = data['DERSLİK'].astype(str).str.extract(r'(\d{2})$').astype(str)

    data['Large_Stairs'] = data['Number_of_People'] * data['ROOM_END'].isin(['01', '02', '06']).astype(int)
    data['Small_Stairs'] = data['Number_of_People'] * data['ROOM_END'].isin(['03', '04', '05']).astype(int)

    stair_usage_per_floor = data.groupby(['Year', 'DAY', 'FLOOR', 'TIMESLOT']).agg({
        'Large_Stairs': 'sum',
        'Small_Stairs': 'sum'
    }).reset_index()

    # Apply redistribution logic on a per-floor basis
    for _, row in stair_usage_per_floor.iterrows():
        floor = row['FLOOR']
        timeslot = row['TIMESLOT']
        day = row['DAY']

        large_stairs = row['Large_Stairs']
        small_stairs = row['Small_Stairs']

        # Check for redistribution conditions (Large -> Small)
        if large_stairs >= 2 * small_stairs:
            # Redistribute people from rooms ending in '02' to small stairs
            mask = (
                    (data['FLOOR'] == floor) &
                    (data['TIMESLOT'] == timeslot) &
                    (data['DAY'] == day) &
                    (data['ROOM_END'] == '02')
            )
            data.loc[mask, 'Small_Stairs'] += data.loc[mask, 'Number_of_People']
            data.loc[mask, 'Large_Stairs'] -= data.loc[mask, 'Number_of_People']

        # Check for redistribution conditions (Small -> Large)
        elif small_stairs >= 2 * large_stairs:
            mask = (
                (data['FLOOR'] == floor) &
                (data['TIMESLOT'] == timeslot) &
                (data['DAY'] == day) &
                (data['ROOM_END'] == '04')
            )
            data.loc[mask, 'Large_Stairs'] += data.loc[mask, 'Number_of_People']
            data.loc[mask, 'Small_Stairs'] -= data.loc[mask, 'Number_of_People']

    stair_usage_total = data.groupby(['Year', 'DAY', 'TIMESLOT']).agg({
        'Large_Stairs': 'sum',
        'Small_Stairs': 'sum'
    }).reset_index()

    # Check for redistribution across all floors
    for _, row in stair_usage_total.iterrows():
        timeslot = row['TIMESLOT']
        day = row['DAY']

        total_large_stairs = row['Large_Stairs']
        total_small_stairs = row['Small_Stairs']

        # Check for redistribution conditions (Large -> Small)
        if total_large_stairs >= 6 * total_small_stairs:
            # Redistribute people from rooms ending in '02' to small stairs
            mask = (
                (data['TIMESLOT'] == timeslot) &
                (data['DAY'] == day) &
                (data['ROOM_END'] == '02')
            )
            data.loc[mask, 'Small_Stairs'] += data.loc[mask, 'Number_of_People']
            data.loc[mask, 'Large_Stairs'] -= data.loc[mask, 'Number_of_People']

        # Check for redistribution conditions (Small -> Large)
        elif total_small_stairs >= 6 * total_large_stairs:
            mask = (
                (data['TIMESLOT'] == timeslot) &
                (data['DAY'] == day) &
                (data['ROOM_END'] == '04')
            )
            data.loc[mask, 'Large_Stairs'] += data.loc[mask, 'Number_of_People']
            data.loc[mask, 'Small_Stairs'] -= data.loc[mask, 'Number_of_People']

    # Replace negative stair usage values with 0
    data['Large_Stairs'] = data['Large_Stairs'].clip(lower=0)
    data['Small_Stairs'] = data['Small_Stairs'].clip(lower=0)

    # Recalculate stair usage after redistribution
    stair_usage_per_floor = data.groupby(['Year', 'DAY', 'FLOOR', 'TIMESLOT']).agg({
        'Large_Stairs': 'sum',
        'Small_Stairs': 'sum'
    }).reset_index()

    stair_usage_total = data.groupby(['Year', 'DAY', 'TIMESLOT']).agg({
        'Large_Stairs': 'sum',
        'Small_Stairs': 'sum'
    }).reset_index()

    stair_usage_summary = pd.merge(stair_usage_per_floor, stair_usage_total,
                                   on=['Year', 'DAY', 'TIMESLOT'], suffixes=('_Floor', '_Total'))

    return stair_usage_summary


def simulate_stair_usage_with_predictions(data_2025):
    if 'Predicted_Random_Forest' not in data_2025.columns:
        raise ValueError("Predicted values are missing in the 2025 data.")

    data_2025['Number_of_People'] = data_2025['Predicted_Random_Forest']

    return simulate_stair_usage(data_2025)


def calculate_stair_density_for_2025_floor_1(stair_usage_2025):
    filtered_data = stair_usage_2025[
        (stair_usage_2025['DAY'] == 'Pazartesi') &
        (stair_usage_2025['TIMESLOT'] == '10:20') &
        (stair_usage_2025['FLOOR'] == 1)
    ]

    if filtered_data.empty:
        return f"No data found for floor 1 on Monday at 10:20 for 2025 predictions."

    large_stairs = filtered_data['Large_Stairs_Floor'].sum()
    small_stairs = filtered_data['Small_Stairs_Floor'].sum()

    large_stairs_density = large_stairs / 20.7
    small_stairs_density = small_stairs / 18.0

    return {
        'Large Stairs Density (Floor 1, 2025)': large_stairs_density,
        'Small Stairs Density (Floor 1, 2025)': small_stairs_density
    }


def calculate_stair_speed(densities):
    speeds = {}
    for stair_type, density in densities.items():
        speed = 1.2 - (0.06 * density) - 0.45
        speeds[f"Speed ({stair_type})"] = speed
    return speeds


def calculate_evacuation_time_for_floor_1(stair_densities, stair_speeds, stair_usage):
    """
    Calculate evacuation time for floor 1 on Mondays at 10:20.
    """
    # Extract densities
    large_stairs_density = stair_densities['Large Stairs Density (Floor 1, 2025)']
    small_stairs_density = stair_densities['Small Stairs Density (Floor 1, 2025)']

    # Extract speeds
    speed_large_stair = stair_speeds['Speed (Large Stairs Density (Floor 1, 2025))']
    speed_small_stair = stair_speeds['Speed (Small Stairs Density (Floor 1, 2025))']

    # Extract the number of people on each type of stair
    number_on_large_stair = stair_usage["Large_Stairs_Floor"]
    number_on_small_stair = stair_usage["Small_Stairs_Floor"]

    # Calculate evacuation times
    time_large_stair = (9 / speed_large_stair) * (number_on_large_stair / (2.3 * large_stairs_density))
    time_small_stair = (9 / speed_small_stair) * (number_on_small_stair / (2 * small_stairs_density))

    # Calculate total evacuation time
    total_evacuation_time = max(time_large_stair.mean(), time_small_stair.mean())
    return total_evacuation_time


if __name__ == "__main__":
    file_paths = [
        "2007.xlsx",
        "2008.xlsx",
        "2009.xlsx",
        "2010.xlsx",
        "2011.xlsx",
        "2012.xlsx",
        "2013.xlsx",
        "2014.xlsx",
        "2015.xlsx",
        "2016.xlsx",
        "2017.xlsx",
        "2018.xlsx",
        "2019.xlsx",
        "2021.xlsx",
        "2022.xlsx",
        "2023.xlsx",
        "2024.xlsx"
    ]

    data = load_and_process_data(file_paths)

    data['TIMESLOT_NUM'] = data['TIMESLOT'].apply(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))
    X = data[['Year', 'TIMESLOT_NUM', 'FLOOR']]
    y = data['Number_of_People']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    models, scaler = train_and_evaluate_models(data)

    data_2025 = predict_2025_rf(data, rf_model, scaler)

    data_2024 = data[data['Year'] == 2024].copy()

    predictions_2024, mse_results_2024 = predict_2024(models, scaler, data_2024)

    save_predictions_to_excel(predictions_2024, mse_results_2024, file_name="2024_predictions_comparison.xlsx")

    data_2025.to_excel("predictions_2025_rf.xlsx", index=False)
    print("Predictions saved to predictions_2025_rf.xlsx")

    stair_usage = simulate_stair_usage(data)

    stair_usage.to_excel("stair_usage_summary.xlsx", index=False)
    print("Stair usage summary saved to stair_usage_summary.xlsx")

    stair_usage = simulate_stair_usage(data)

    stair_usage_2025 = simulate_stair_usage_with_predictions(data_2025)

    densities_2025_floor_1 = calculate_stair_density_for_2025_floor_1(stair_usage_2025)

    # Calculate stair speeds using the densities
    stair_speeds_2025 = calculate_stair_speed(densities_2025_floor_1)

    print("\nStair Speeds for Floor 1 on Monday at 10:20 (2025 Predictions):")
    print(stair_speeds_2025)

    # Display the results
    print("Stair Densities for Floor 1 on Monday at 10:20 (2025 Predictions):")
    print(densities_2025_floor_1)

    evacuation_time = calculate_evacuation_time_for_floor_1(densities_2025_floor_1, stair_speeds_2025, stair_usage_2025)

    print("\nTotal Evacuation Time for Floor 1 on Monday at 10:20 (2025 Predictions):")
    print(evacuation_time)