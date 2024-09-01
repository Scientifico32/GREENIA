import openai
import os
import numpy as np
import pandas as pd


# Placeholder for OpenAI client setup (replace with your API key)
client = openai.Client(api_key="YOUR-KEY")

# Aggregate information from all 30 days
combined_optimized_features = []

for best_individual in best_individuals:
    # Truncate or pad the best_individual to match the length of feature_names
    if len(best_individual) < len(feature_names):
        best_individual_truncated = np.pad(best_individual, (0, len(feature_names) - len(best_individual)), 'constant')
    else:
        best_individual_truncated = best_individual[:len(feature_names)]

    # Collect the combined optimized features for summary
    combined_optimized_features.append(best_individual_truncated)

# Average the combined optimized features over all 30 days
avg_optimized_features = np.mean(combined_optimized_features, axis=0)

# Format the averaged features into a readable string
optimized_features_summary = ', '.join(
    [f"{feature_names[i]}: {avg_optimized_features[i]:.3f}" for i in range(len(feature_names))]
)

# Prepare the SHAP summary from the SHAP table (top features across all days)
shap_summary = shap_table.head(10).to_string(index=False)  # Display top 10 SHAP values for brevity

# Construct the prompt text in Greek (or your preferred language)
prompt_text = (
    "Based on the SHAP values and optimal features found through the optimization process over 30 days, "
    "provide a comprehensive strategic plan for the placement of renewable energy sources (RES) across Greece. "
    "The SHAP values represent the most influential factors affecting renewable energy production, and the optimal "
    "features indicate the best conditions for maximizing energy output. Consider the following top SHAP values:\n" 
    + shap_summary + "\n\n"
    "Additionally, the average optimal conditions for energy production across all days are summarized as follows: " 
    + optimized_features_summary + ".\n\n"
    "Based on these insights, provide a detailed strategic plan for the optimal placement of different types of RES "
    "across various geographical locations in Greece. The plan should focus on maximizing energy production efficiency "
    "and sustainability."
)

# The request to OpenAI's API remains the same
completion = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt=prompt_text,
    max_tokens=2000,
    temperature=0
)

# Extract the strategic plan from the response
strategic_plan = completion.choices[0].text

# Save the strategic plan to a text file
with open("strategic_plan.txt", "w") as text_file:
    text_file.write("Strategic Planning for Renewable Energy Deployment in Greece:\n\n")
    text_file.write(strategic_plan)

# Optional: Print confirmation message
print("Strategic plan has been saved to 'strategic_plan.txt'.")
