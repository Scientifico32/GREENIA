import numpy as np
import torch
import joblib
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from model_definition import SimpleGRU, SimpleLSTM  # Import models directly

def pso(objective_function, bounds, num_particles=50, max_iter=1000):
    dim = len(bounds)
    swarm = []

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

        for particle in swarm:
            inertia = 0.5
            cognitive = 1.5 * np.random.rand(dim) * (particle['best_position'] - particle['position'])
            social = 1.5 * np.random.rand(dim) * (global_best_position - particle['position'])
            particle['velocity'] = inertia * particle['velocity'] + cognitive + social
            particle['position'] += particle['velocity']
            particle['position'] = np.clip(particle['position'], [b[0] for b in bounds], [b[1] for b in bounds])

    return global_best_position

def run_optimization(n_days, X_test_tensor, y_test_tensor, gru_path, lstm_path, xgb_path, scaler_y_path, feature_dim=50, seq_len=10):
    # Load models directly in the function
    model_gru = SimpleGRU(input_dim=5, hidden_dim=128, output_dim=1, seq_len=10)
    model_gru.load_state_dict(torch.load(gru_path))
    model_gru.eval()

    model_lstm = SimpleLSTM(input_dim=5, hidden_dim=128, output_dim=1, seq_len=10)
    model_lstm.load_state_dict(torch.load(lstm_path))
    model_lstm.eval()

    xgb_model = joblib.load(xgb_path)
    scaler_y = joblib.load(scaler_y_path)

    bounds = [(0, 1) for _ in range(feature_dim)]
    best_individuals = []
    actual_values = []
    forecasted_values = []
    optimized_values = []
    improvements = []

    for day in range(n_days):
        print(f"Optimizing for Day {day + 1}...")

        def objective_function(individual):
            input_tensor = torch.tensor(np.array(individual).reshape(1, seq_len, feature_dim // seq_len), dtype=torch.float32)
            prediction_gru = model_gru(input_tensor).item()
            prediction_lstm = model_lstm(input_tensor).item()
            input_flat = input_tensor.view(1, -1).numpy()
            prediction_xgb = xgb_model.predict(input_flat)[0]
            ensemble_prediction = (prediction_gru + prediction_lstm + prediction_xgb) / 3
            return -ensemble_prediction

        best_individual = pso(objective_function, bounds, num_particles=30, max_iter=1000)

        if len(best_individual) < 50:
            best_individual_padded = np.pad(best_individual, (0, 50 - len(best_individual)), 'constant')
        elif len(best_individual) > 50:
            best_individual_padded = best_individual[:50]
        else:
            best_individual_padded = best_individual

        best_individual_original_space = best_individual_padded[:X_test_tensor.shape[1]]
        best_individual_orig = scaler_X.inverse_transform(np.array(best_individual_original_space).reshape(1, -1)).flatten()
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

        print(f"Best Individual for Day {day + 1} (Original Scale): {best_individual_orig}")
        print(f"Forecasted Energy Production for Day {day + 1} (Original Scale, before optimization): {forecasted_value_orig}")
        print(f"Forecasted Energy Production for Day {day + 1} (Original Scale, after PSO optimization): {forecasted_value_optimized_orig}")
        print(f"Improvement for Day {day + 1}: {improvement}")
        print(f"Percentage Improvement for Day {day + 1}: {percentage_improvement:.2f}%")

    plot_results(n_days, actual_values, optimized_values, improvements)

def plot_results(n_days, actual_values, optimized_values, improvements):
    days = np.arange(1, n_days + 1)

    plt.figure(figsize=(12, 8))
    plt.plot(days, actual_values, label='Actual Forecasted Values (Before Optimization)', marker='o', linestyle='--')
    plt.plot(days, optimized_values, label='Forecasted Values (After Optimization)', marker='o', linestyle='--')
    plt.bar(days, improvements, label='Improvement', alpha=0.3)
    plt.xlabel('Day')
    plt.ylabel('Energy Production (MWh)')
    plt.title('Energy Production: Actual vs Optimized')
    plt.legend()
    plt.grid(True)
    plt.show()

