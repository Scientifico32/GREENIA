import pandas as pd
import numpy as np

def load_data():
    # Load and clean all your data files
    dfs = []
    filenames = ['AGCHIALOS_16665.xlsx', 'AKTIO_16643.xlsx', 'ALEXANDROUPOLI_16627.xlsx',
                 'ARAXOS_16687.xlsx', 'ARGOS_16724.xlsx', 'FLORINA_16613.xlsx',
                 'IOANNINA_16642.xlsx', 'KALAMATA_16726.xlsx', 'KAVALA_16624.xlsx', 'KOZANI_16632.xlsx']
    for file in filenames:
        df = pd.read_excel(f'/content/drive/MyDrive/EMY/{file}', skiprows=9)
        df.columns = list(df.columns[:4]) + [f'{col}_{file.split("_")[0].lower()}' for col in df.columns[4:]]
        dfs.append(df)
    return dfs

def merge_data(dfs):
    common_columns = ["Έτος", "Μήνας", "Μέρα", "Ώρα (UTC)"]
    final_df = dfs[0]
    for df in dfs[1:]:
        final_df = final_df.merge(df.drop(columns=common_columns), left_index=True, right_index=True, suffixes=('', '_y'))

    final_df['Date'] = pd.to_datetime(final_df[['Έτος', 'Μήνας', 'Μέρα', 'Ώρα (UTC)']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d-%H')
    final_df.set_index('Date', inplace=True)
    final_df.drop(columns=['Έτος', 'Μήνας', 'Μέρα', 'Ώρα (UTC)'], inplace=True)
    final_df_resampled = final_df.resample('D').first()
    final_df_resampled.reset_index(inplace=True)
    return final_df_resampled

def clean_data(final_df_resampled, energy_file='/content/drive/MyDrive/Book3.csv'):
    df_energy = pd.read_csv(energy_file)
    df_energy = df_energy.iloc[:, [0, -1]]
    final_df_resampled = final_df_resampled.replace(',', '.', regex=True)
    final_df_resampled = final_df_resampled.loc[365:1616, :]
    final_df_resampled_cleaned = final_df_resampled.copy()

    final_df_resampled_cleaned['Date'] = pd.to_datetime(final_df_resampled_cleaned['Date'], format='%Y-%m-%d %H:%M:%S')
    df_energy['Date'] = pd.to_datetime(df_energy['Date'], format='%m/%d/%Y')

    merged_df = final_df_resampled_cleaned.merge(df_energy[['Date', 'Energy (MWh)']], on='Date', how='inner')
    
    # Handle missing data
    cols_to_drop = merged_df.columns[merged_df.isnull().sum() > 65]
    merged_df.drop(columns=cols_to_drop, inplace=True)

    for column in merged_df.columns:
        if merged_df[column].dtype in ['int64', 'float64']:
            nan_indices = merged_df.index[merged_df[column].isnull()]
            for idx in nan_indices:
                prev_value = merged_df[column].loc[:idx].dropna().iloc[-1] if not merged_df[column].loc[:idx].dropna().empty else np.nan
                next_value = merged_df[column].loc[idx:].dropna().iloc[0] if not merged_df[column].loc[idx:].dropna().empty else np.nan
                mean_value = np.nanmean([prev_value, next_value])
                merged_df.at[idx, column] = mean_value

    merged_df.fillna(merged_df.mean(), inplace=True)
    return merged_df

if __name__ == "__main__":
    dfs = load_data()
    final_df_resampled = merge_data(dfs)
    merged_df = clean_data(final_df_resampled)
    merged_df.to_csv('/content/drive/MyDrive/TimeMixWithAttention/processed_data/merged_df.csv', index=False)
    print("Data preprocessing completed and saved.")
