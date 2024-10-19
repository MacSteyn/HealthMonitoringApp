import pandas as pd
import matplotlib.pyplot as plt

# Function to collect health data from the user
def collect_health_data():
    heart_rate = int(input("Enter your heart rate (bpm): "))
    blood_pressure = int(input("Enter your blood pressure (systolic): "))
    body_temperature = float(input("Enter your body temperature (°C): "))
    
    # Return data as a dictionary
    return {
        'Heart Rate': heart_rate,
        'Blood Pressure': blood_pressure,
        'Body Temperature': body_temperature
    }

# Function to save the data to a CSV file
def save_health_data(data, filename='health_data.csv'):
    try:
        # If file exists, load the existing data
        health_data = pd.read_csv(filename)
        print("Existing health data found, appending new data.")
    except FileNotFoundError:
        # If no file exists, create a new DataFrame
        health_data = pd.DataFrame(columns=['Heart Rate', 'Blood Pressure', 'Body Temperature'])
        print("No previous health data found, creating new dataset.")
    
    # Append the new data to the DataFrame
    health_data = health_data._append(data, ignore_index=True)

    # Save the updated DataFrame back to CSV
    health_data.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

# Function to check for anomalies
def check_for_anomalies(data):
    heart_rate = data['Heart Rate']
    blood_pressure = data['Blood Pressure']
    body_temperature = data['Body Temperature']
    
    anomalies = []

    # Simple thresholds for anomaly detection (can be adjusted)
    if not 60 <= heart_rate <= 100:
        anomalies.append(f"Abnormal heart rate: {heart_rate} bpm")
    if not 90 <= blood_pressure <= 120:
        anomalies.append(f"Abnormal blood pressure: {blood_pressure} mmHg")
    if not 36.1 <= body_temperature <= 37.2:
        anomalies.append(f"Abnormal body temperature: {body_temperature} °C")
    
    return anomalies

# Function to plot health data
def plot_health_data(filename='health_data.csv'):
    try:
        health_data = pd.read_csv(filename)
        health_data.plot(subplots=True, figsize=(10, 8), title="Health Monitoring Data")
        plt.show()
    except FileNotFoundError:
        print("No health data found to plot.")

# Main function
def main():
    # Collect new health data
    user_data = collect_health_data()

    # Save the collected data
    save_health_data(user_data)

    # Check for any anomalies in the data
    anomalies = check_for_anomalies(user_data)
    if anomalies:
        print("Anomalies detected:")
        for anomaly in anomalies:
            print(f"- {anomaly}")
    else:
        print("No anomalies detected.")

    # Plot the updated health data
    plot_health_data()

if __name__ == "__main__":
    main()
