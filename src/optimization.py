import numpy as np
import numpy as np
import random
import torch
import matplotlib.pyplot as plt

# 1. Jaya Optimization Algorithm
def jaya_algorithm(objective_function, initial_solution, bounds, max_iter=200):
    best_solution = initial_solution.copy()
    best_value = objective_function(best_solution)

    for iteration in range(max_iter):
        for i in range(len(best_solution)):
            new_solution = best_solution.copy()

            candidate_1 = random.uniform(bounds[i][0], bounds[i][1])
            candidate_2 = random.uniform(bounds[i][0], bounds[i][1])

            new_solution[i] = new_solution[i] + random.uniform(0, 1) * (candidate_1 - abs(new_solution[i])) - random.uniform(0, 1) * (candidate_2 - abs(new_solution[i]))
            new_solution[i] = np.clip(new_solution[i], bounds[i][0], bounds[i][1])

        new_value = objective_function(new_solution)

        if new_value < best_value:
            best_solution = new_solution
            best_value = new_value

    return best_solution

# 2. Running the Optimization and Printing Results
def run_optimization(model, X_test_tensor, y_test_tensor, scaler_X, scaler_y, seq_len=10, feature_dim=50, n_days=30):
    # Calculate bounds based on the minimum and maximum values of each input feature
    bounds = [(np.min(X_test_tensor[:, :, i].numpy()), np.max(X_test_tensor[:, :, i].numpy())) for i in range(X_test_tensor.shape[2])]
    
    actual_values = []
    forecasted_values = []
    optimized_values = []
    improvements = []

    for day in range(n_days):
        print(f"Optimizing for Day {day + 1}...")

        def objective_function(individual):
            input_tensor = torch.tensor(np.array(individual).reshape(1, seq_len, feature_dim // seq_len), dtype=torch.float32)
            forecasted_value = model(input_tensor).item()
            return -forecasted_value

        best_individual = jaya_algorithm(objective_function, X_test_tensor[day].flatten().numpy(), bounds, max_iter=1000)

        if len(best_individual) < 50:
            best_individual_padded = np.pad(best_individual, (0, 50 - len(best_individual)), 'constant')
        elif len(best_individual) > 50:
            best_individual_padded = best_individual[:50]
        else:
            best_individual_padded = best_individual

        best_individual_original_space = best_individual_padded[:X_test_tensor.shape[2]]

        best_individual_orig = scaler_X.inverse_transform(np.array(best_individual_original_space).reshape(1, -1)).flatten()

        input_tensor = torch.tensor(np.array(best_individual_padded).reshape(1, seq_len, feature_dim // seq_len), dtype=torch.float32)
        forecasted_value_optimized = model(input_tensor).item()

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
        print(f"Forecasted Energy Production for Day {day + 1} (Original Scale, after Jaya optimization): {forecasted_value_optimized_orig}")
        print(f"Improvement for Day {day + 1}: {improvement}")
        print(f"Percentage Improvement for Day {day + 1}: {percentage_improvement:.2f}%")

    # 3. Plotting Results
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

# Example usage:
# Assuming model, X_test_tensor, y_test_tensor, scaler_X, and scaler_y have been defined
# run_optimization(model, X_test_tensor, y_test_tensor, scaler_X, scaler_y)


