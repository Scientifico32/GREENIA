import numpy as np
import random
import torch
from sklearn.preprocessing import MinMaxScaler
from model_training import model, scaler_X, scaler_y
from model_definition import TimeMixWithAttention
import matplotlib.pyplot as plt

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

n_days = 7
bounds = [(0, 1) for _ in range(50)]
best_individuals = []
actual_values = []
forecasted_values = []
optimized_values = []
improvements = []

for day in range(n_days):
    print(f"Optimizing for Day {day + 1}...")

    def objective_function(individual):
        input_tensor = torch.tensor(np.array(individual).reshape(1, 10, 50 // 10), dtype=torch.float32)
        forecasted_value = model(input_tensor).item()
        return -forecasted_value

    best_individual = jaya_algorithm(objective_function, X_test_tensor[day].flatten().numpy(), bounds, max_iter=1000)

    best_individual_padded = np.pad(best_individual, (0, 50 - len(best_individual)), 'constant') if len(best_individual) < 50 else best_individual[:50]
    best_individual_original_space = best_individual_padded[:X.shape[1]]
    best_individual_orig = scaler_X.inverse_transform(np.array(best_individual_original_space).reshape(1, -1)).flatten()

    best_individuals.append(best_individual_orig)

    input_tensor = torch.tensor(np.array(best_individual_padded).reshape(1, 10, 50 // 10), dtype=torch.float32)
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
