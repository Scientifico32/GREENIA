import openai
import os
import numpy as np
import pandas as pd

# Load saved SHAP values and feature names
shap_table = pd.read_csv('/content/drive/MyDrive/TimeMixWithAttention/shap_table.csv')
best_individuals = np.load('/content/drive/MyDrive/TimeMixWithAttention/best_individuals.npy', allow_pickle=True)

client = openai.Client(api_key="your-openai-api-key")

strategic_recommendations = []

for day, best_individual in enumerate(best_individuals, start=1):
    best_individual_truncated = best_individual[:len(shap_table['Features'])]

    optimized_features_summary = ', '.join(
        [f"{shap_table['Features'][i]}: {best_individual_truncated[i]:.3f}" for i in range(len(shap_table['Features']))]
    )

    shap_summary = shap_table.head(10).to_string(index=False)

    prompt_text = (
        "Interpret the following SHAP values comprehensively for renewable energy production companies. "
        "Explain how these values influence the prediction of the Deep Learning model and how these results help "
        "optimize the placement of renewable energy sources, referring to geographical locations in Greece. "
        "Consider the following top SHAP values:\n" + shap_summary + "\n\n"
        "Additionally, consider the optimal set of features found through optimization, which suggests optimal "
        "conditions for energy production: " + optimized_features_summary + ".\n\n"
        "Provide a detailed and extensive explanation based on the above information, by combining all the information for each day in one weekly report that suggests where to place the different kind of RES in order to maximize the energy:"
    )

    completion = client.create_chat_completion(
      model="gpt-3.5-turbo-instruct",
      messages=[{"role": "user", "content": prompt_text}],
      max_tokens=2000,
      temperature=0
    )

    explanation = completion['choices'][0]['message']['content']
    print(f"Explanation for Day {day}:\n", explanation, "\n")
    strategic_recommendations.append(f"Day {day} - Strategic Insights:\n" + explanation + "\n")

strategic_plan = "\n".join(strategic_recommendations)
save_dir = '/content/drive/MyDrive/TimeMixWithAttention'
strategic_plan_path = os.path.join(save_dir, 'strategic_plan.txt')

with open(strategic_plan_path, 'w') as f:
    f.write(strategic_plan)
print("Strategic plan saved to:", strategic_plan_path)
