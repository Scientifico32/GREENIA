# GREENIA

🌍 Renewable Energy Forecasting and Strategic Planning 🌍

📊 Welcome to the Renewable Energy Forecasting and Strategic Planning project! This repository contains the full implementation of a deep learning model designed to forecast energy production based on meteorological data. The project also includes strategic planning for optimizing the placement of renewable energy sources (RES) across Greece 📊

🗂️ Project Structure

/project-directory/
  │
  ├── /data/
  │   ├── raw_data/
  │   │   ├── AGCHIALOS_16665.xlsx
  │   │   ├── AKTIO_16643.xlsx
  │   │   ├── ALEXANDROUPOLI_16627.xlsx
  │   │   └── ... (other Excel files)
  │   └── processed_data/
  │       └── final_df_resampled_cleaned.csv
  │
  ├── /src/
  │   ├── data_preprocessing.py
  │   ├── model_definition.py
  │   ├── model_training.py
  │   ├── optimization.py
  │   ├── shap_analysis.py
  │   ├── strategic_planning.py
  │   └── utils.py
  │
  ├── /models/
  │   ├── timemix_with_attention_model.pth
  │   ├── scaler_X.pkl
  │   ├── scaler_y.pkl
  │   ├── shap_explainer.pkl
  │   └── shap_values.pkl
  │
  ├── /notebooks/
  │   └── analysis_notebook.ipynb
  │
  ├── /outputs/
  │   ├── training_log.csv
  │   └── strategic_plan.txt
  │
  ├── README.md
  ├── requirements.txt
  └── .gitignore

📁 /data/
- raw_data/: Contains the raw meteorological data from various locations in Greece.
- processed_data/: Stores the processed and cleaned data ready for modeling.

🧠 /src/
- data_preprocessing.py: Scripts for data cleaning, feature engineering, and preprocessing.
- model_definition.py: Contains the deep learning model architecture, including the TimeMix model with an attention mechanism.
- model_training.py: Handles the training process of the model, including evaluation metrics.
- optimization.py: Implements the Jaya and Genetic Algorithms for optimizing the input features.
- shap_analysis.py: Performs SHAP analysis to interpret model predictions and identify the most influential features.
- strategic_planning.py: Utilizes the SHAP analysis results to generate strategic recommendations for RES placement.
- utils.py: Contains utility functions for data handling, model evaluation, and more.

🗄️ /models/
- timemix_with_attention_model.pth: The trained TimeMix model with attention mechanism.
- scaler_X.pkl: Scaler object used for feature scaling.
- scaler_y.pkl: Scaler object used for target scaling.
- shap_explainer.pkl: Saved SHAP explainer object.
- shap_values.pkl: Saved SHAP values for model interpretation.

📓 /notebooks/
- analysis_notebook.ipynb: Jupyter notebook containing exploratory data analysis, model training steps, and visualization.

📤 /outputs/
training_log.csv: A log of the training process, including loss per epoch.
strategic_plan.txt: The final strategic plan generated for optimal RES placement.

🚀 Getting Started
Prerequisites
Python 3.7+
Git
Required Python packages (listed in requirements.txt)

Installation

Clone the Repository:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
