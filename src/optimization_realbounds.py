import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Assuming your dataset is loaded into merged_df
X = merged_df.drop(columns=['Energy (MWh)', 'Date']).values
y = merged_df['Energy (MWh)'].values

# Scale the features and the target variable
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Padding and Splitting the Data
X_padded = np.pad(X_scaled, ((0, 0), (0, 3)), 'constant')
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_scaled, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

seq_len = 10
feature_dim = 50  # Padded to 50
X_train_tensor = X_train_tensor.view(-1, seq_len, feature_dim // seq_len)
X_test_tensor = X_test_tensor.view(-1, seq_len, feature_dim // seq_len)

# Calculate the bounds based on the original data
min_values = X.min(axis=0)  # Minimum values for each feature
max_values = X.max(axis=0)  # Maximum values for each feature
bounds = [(min_val, max_val) for min_val, max_val in zip(min_values, max_values)]

# Load the models (Adjust paths and models as necessary)
model_gru = SimpleGRU(input_dim=5, hidden_dim=128, output_dim=1, seq_len=10)
model_gru.load_state_dict(torch.load('/models/gru_model.pth'))
model_gru.eval()

model_lstm = SimpleLSTM(input_dim=5, hidden_dim=128, output_dim=1, seq_len=10)
model_lstm.load_state_dict(torch.load('/models/lstm_model.pth'))
model_lstm.eval()

xgb_model = joblib.load('/models/xgboost_model.pkl')
scaler_y = joblib.load('/models/scaler_y.pkl')

# Particle Swarm Optimization (PSO)
def pso(objective_function, bounds, num_particles=10, max_iter=100):
    dim = len(bounds)
    swarm = []

    # Initialize the swarm
    for _ in range(num_particles):
        particle = {
            'position': np.array([random.uniform(b[0], b[1]) for b in bounds]),
            'velocity': np.random.uniform(-1, 1, dim),
            'best_position': None,
            'best_value': float('inf')
        }
        particle['best_position'] = particle['position'].copy()
        swarm.append(particle)

    global_best_position = None
    global_best_value = float('inf')

    for iteration in range(max_iter):
        for particle in swarm:
            value = objective_function(particle['position'])

            if value < particle['best_value']:
                particle['best_value'] = value
                particle['best_position'] = particle['position'].copy()

            if value < global_best_value:
                global_best_value = value
                global_best_position = particle['position'].copy()

        # Update the velocity and position of each particle
        for particle in swarm:
            inertia = 0.5
            cognitive = 1.5 * np.random.rand(dim) * (particle['best_position'] - particle['position'])
            social = 1.5 * np.random.rand(dim) * (global_best_position - particle['position'])
            particle['velocity'] = inertia * particle['velocity'] + cognitive + social
            particle['position'] += particle['velocity']
            particle['position'] = np.clip(particle['position'], [b[0] for b in bounds], [b[1] for b in bounds])

    return global_best_position

# Running the Optimization and Collecting Results
n_days = 7

best_individuals = []  # Initialize a list to store best individuals for each day
actual_values = []
forecasted_values = []
optimized_values = []
improvements = []

# Open a file to save the outputs
with open('optimization_results.txt', 'w') as file:

    for day in range(n_days):
        print(f"Optimizing for Day {day + 1}...")
        file.write(f"Optimizing for Day {day + 1}...\n")

        def objective_function(individual):
            input_tensor = torch.tensor(np.array(individual).reshape(1, seq_len, feature_dim // seq_len), dtype=torch.float32)
            
            # Get predictions from the GRU and LSTM models
            prediction_gru = model_gru(input_tensor).item()
            prediction_lstm = model_lstm(input_tensor).item()

            # Flatten the input for XGBoost (2D instead of 3D tensor)
            input_flat = input_tensor.view(1, -1).numpy()
            prediction_xgb = xgb_model.predict(input_flat)[0]

            # Ensemble method: Average the predictions
            ensemble_prediction = (prediction_gru + prediction_lstm + prediction_xgb) / 3
            
            # We negate the ensemble prediction because PSO is designed to minimize the objective function
            return -ensemble_prediction

        best_individual = pso(objective_function, bounds, num_particles=30, max_iter=1000)

        if len(best_individual) < 50:
            best_individual_padded = np.pad(best_individual, (0, 50 - len(best_individual)), 'constant')
        elif len(best_individual) > 50:
            best_individual_padded = best_individual[:50]
        else:
            best_individual_padded = best_individual

        best_individual_original_space = best_individual_padded[:X.shape[1]]

        best_individual_orig = scaler_X.inverse_transform(np.array(best_individual_original_space).reshape(1, -1)).flatten()

        # Store the best individual in the list
        best_individuals.append(best_individual_orig)

        input_tensor = torch.tensor(np.array(best_individual_padded).reshape(1, seq_len, feature_dim // seq_len), dtype=torch.float32)
        forecasted_value_optimized = model_gru(input_tensor).item()

        forecasted_value_optimized_orig = scaler_y.inverse_transform([[forecasted_value_optimized]]).flatten()[0]

        forecasted_value_orig = scaler_y.inverse_transform([[y_test_tensor[day].item()]]).flatten()[0]

        improvement = forecasted_value_optimized_orig - forecasted_value_orig
        percentage_improvement = (improvement / forecasted_value_orig) * 100

        actual_values.append(forecasted_value_orig)
        forecasted_values.append(forecasted_value_orig)
        optimized_values.append(forecasted_value_optimized_orig)
        improvements.append(improvement)

        # Write the results to the file
        file.write(f"Best Individual for Day {day + 1} (Original Scale): {best_individual_orig}\n")
        file.write(f"Forecasted Energy Production for Day {day + 1} (Original Scale, before optimization): {forecasted_value_orig}\n")
        file.write(f"Forecasted Energy Production for Day {day + 1} (Original Scale, after PSO optimization): {forecasted_value_optimized_orig}\n")
        file.write(f"Improvement for Day {day + 1}: {improvement}\n")
        file.write(f"Percentage Improvement for Day {day + 1}: {percentage_improvement:.2f}%\n")
        file.write("\n")

# Plotting Results
days = np.arange(1, n_days + 1)

plt.figure(figsize=(12, 8))

# Plot the forecasted values
plt.plot(days, actual_values, label='Actual Forecasted Values (Before Optimization)', marker='o', linestyle='--')
plt.plot(days, optimized_values, label='Forecasted Values (After Optimization)', marker='o', linestyle='--')

# Plot the improvements
plt.bar(days, improvements, label='Improvement', alpha=0.3)

plt.xlabel('Day')
plt.ylabel('Energy Production (MWh)')
plt.title('Energy Production: Actual vs Optimized')
plt.legend()
plt.grid(True)
plt.show()

# Save the plot to a file
plt.savefig('optimization_plot.png')

# Closing the file (not strictly necessary with the 'with' statement, but good practice)
file.close()
