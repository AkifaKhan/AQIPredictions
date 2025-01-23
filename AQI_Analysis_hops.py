import requests
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import hopsworks # type: ignore
import joblib

# Function to fetch air pollution data
def fetch_air_pollution_data(lat, lon, start, end, api_key):
    url = "http://api.openweathermap.org/data/2.5/air_pollution/history"

    params = {
        "lat": lat,
        "lon": lon,
        "start": start,
        "end": end,
        "appid": api_key
    }
    response = requests.get(url, params=params)
    api_response = response.json()
    
    data = [
        {
            'datetime': datetime.utcfromtimestamp(item['dt']).strftime('%Y-%m-%d %H:%M:%S'),
            'aqi': item['main']['aqi'],
            **item['components']
        }
        for item in api_response['list']
    ]
    
    return pd.DataFrame(data)

# Function to fetch weather data
def fetch_monthly_data(lat, lon, start_date, end_date, api_key):
    base_url = "https://api.weatherbit.io/v2.0/history/hourly"
    params = {
        "lat": lat,
        "lon": lon,
        "start_date": start_date,
        "end_date": end_date,
        "key": api_key
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json().get('data', [])
    else:
        print(f"Error fetching data: {response.status_code}, {response.text}")
        return []

# Main function to execute the analysis
def main():
    # Hopsworks login
    project_name = "AQIPredictions"
    api_key = "b5GR3KdA1sCXrxzX.W29cSj9sMnypht8dLG4FYgjxebpKQxzbQPvOZPWpz5IyEko23UIXGiwAcuFENXfF"
    connection = hopsworks.login(api_key=api_key)
    project = connection.get_project(project_name)

    # Air pollution parameters
    api_key_pollution = "b219bcab2ed1426432de1ff7c768e46a"
    lat, lon = 33.44, -94.04
    start_timestamp = 1704067200  # Jan 1, 2024
    end_timestamp = 1735689600    # Dec 31, 2024

    # Fetch air pollution data
    data_air_pollution = fetch_air_pollution_data(lat, lon, start_timestamp, end_timestamp, api_key_pollution)
    data_air_pollution['datetime'] = pd.to_datetime(data_air_pollution['datetime'])
    data_air_pollution['day'] = data_air_pollution['datetime'].dt.day
    data_air_pollution['month'] = data_air_pollution['datetime'].dt.month
    data_air_pollution['date'] = data_air_pollution['datetime'].dt.date

    # Group by date and calculate the mean for numeric columns
    daily_data = data_air_pollution.groupby('date').mean().reset_index()
    daily_data['day'] = pd.to_datetime(daily_data['date']).dt.day
    daily_data['month'] = pd.to_datetime(daily_data['date']).dt.month

    # Weather data parameters
    api_key_weather = "a7475fbd2cb84f8ca148613df19e1ba4"
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)

    # Fetch weather data
    current_date = start_date
    all_data = []
    while current_date < end_date:
        start_str = current_date.strftime('%Y-%m-%d')
        next_month = (current_date + timedelta(days=32)).replace(day=1)
        end_str = (next_month - timedelta(days=1)).strftime('%Y-%m-%d')
        
        print(f"Fetching data from {start_str} to {end_str}")
        monthly_data = fetch_monthly_data(lat, lon, start_str, end_str, api_key_weather)
        all_data.extend(monthly_data)
        
        current_date = next_month

    # Convert all data to a single DataFrame
    data_weather = pd.DataFrame(all_data)
    data_weather['datetime'] = data_weather['datetime'].str.replace(':', ' ')
    data_weather['datetime'] = pd.to_datetime(data_weather['datetime'])

    # Merge datasets
    merged_dataset = pd.merge(data_air_pollution, data_weather, on='datetime', how='inner')
    merged_dataset = merged_dataset.drop(columns=['h_angle', 'datetime', 'date', 'timestamp_local', 'timestamp_utc', 'pod', 'revision_status', 'weather'])

    # Prepare data for modeling
    X = merged_dataset.drop('aqi', axis=1)
    y = merged_dataset['aqi']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")

    # Predict the next three days
    future_dates = [data_air_pollution['datetime'].max() + timedelta(days=i) for i in range(1, 4)]
    future_data = pd.DataFrame(future_dates, columns=['datetime'])
    
    # Create features for future dates
    future_data['day'] = future_data['datetime'].dt.day
    future_data['month'] = future_data['datetime'].dt.month
    future_data['year'] = future_data['datetime'].dt.year
    future_data['day_of_week'] = future_data['datetime'].dt.dayofweek
    future_data['hour'] = 0  # Assuming we want to predict for the start of the day

    # Simulate or estimate values for missing features
    mean_values = X_train.mean()
    for col in mean_values.index:
        if col not in future_data.columns:
            future_data[col] = mean_values[col]

    # Ensure the future data has the same features as the training data
    future_data = future_data[X_train.columns]  # Align future data with training features

    # Make predictions for the next three days
    future_predictions = model.predict(future_data)
    
    # Display predictions
    for i, date in enumerate(future_dates):
        print(f"Predicted AQI for {date.date()}: {future_predictions[i]}")

    # Save and register the model
    model_path = "path/to/your/model/aqi_model.joblib"
    joblib.dump(model, model_path)

    # Register the model in Hopsworks
    model_name = "AQI_Prediction_Model"
    model_version = "1.0"
    project.get_model_registry().register(model_name=model_name, model_version=model_version, model_path=model_path)

    # Register features in Hopsworks Feature Store
    feature_group = project.get_feature_group("aqipredictions")
    feature_group.insert(data_air_pollution)

if __name__ == "__main__":
    main()