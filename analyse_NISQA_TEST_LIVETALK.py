# %%
# Load and prepare female and male raw data for each listener across all dimensions
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t, shapiro, wilcoxon

raw_data_path = "./NISQA_TEST_LIVETALK_listening_test_ratings.csv"
raw_df = pd.read_csv(raw_data_path, sep=";", encoding="latin-1")
raw_df.rename(columns={'Condition': 'File'}, inplace=True)

def prep_raw_data(raw_df, dim):
    unique_files = raw_df['File'].unique()
    df = pd.DataFrame({'File': unique_files})
    pivot_df = raw_df.pivot(index='File', columns='PID', values=dim)
    result_df = df.merge(pivot_df, on='File', how='left')

    female_raw_df, male_raw_df = [
        result_df[result_df['File'].str.contains(gender, na=False)].sort_values(by='File').reset_index(drop=True)
        for gender in ['f', 'm']
    ]

    female_raw_df['File'] = female_raw_df['File'].str[:3]
    male_raw_df['File'] = male_raw_df['File'].str[:3]

    female_raw_df = female_raw_df.groupby('File').agg(list).reset_index()
    male_raw_df = male_raw_df.groupby('File').agg(list).reset_index()

    cols_to_split = list(range(1, 25))  # Integers 1 to 24

    # For FEMALE
    for col in cols_to_split:
        split_cols = pd.DataFrame(female_raw_df[col].tolist(), index=female_raw_df.index)
        split_cols.columns = [f"{col}_1", f"{col}_2"]
        female_raw_df[[f"{col}_1", f"{col}_2"]] = split_cols

    female_raw_df.drop(columns=cols_to_split, inplace=True)

    # For MALE
    for col in cols_to_split:
        split_cols = pd.DataFrame(male_raw_df[col].tolist(), index=male_raw_df.index)
        split_cols.columns = [f"{col}_1", f"{col}_2"]
        male_raw_df[[f"{col}_1", f"{col}_2"]] = split_cols

    male_raw_df.drop(columns=cols_to_split, inplace=True)

    return female_raw_df, male_raw_df

female_raw_qoe_df, male_raw_qoe_df = prep_raw_data(raw_df, 'QOE')
female_raw_noi_df, male_raw_noi_df = prep_raw_data(raw_df, 'NOI')
female_raw_col_df, male_raw_col_df = prep_raw_data(raw_df, 'COL')
female_raw_dis_df, male_raw_dis_df = prep_raw_data(raw_df, 'DIS')
female_raw_lou_df, male_raw_lou_df = prep_raw_data(raw_df, 'LOU')
female_raw_qoe_df
# %%
# Plot female vs male speaker rating distributions across the dimensions
def plot_distributions(f_df, m_df, dim):
    # Flatten all ratings into a single Series for each gender
    female_values = f_df.drop(columns='File').values.flatten()
    male_values = m_df.drop(columns='File').values.flatten()

    plt.figure(figsize=(10, 6))
    plt.hist(female_values, bins=range(1, 7), alpha=0.5, label='Female', color='green', edgecolor='black')
    plt.hist(male_values, bins=range(1, 7), alpha=0.5, label='Male', color='purple', edgecolor='black')

    plt.title(f"Distribution of Individual {dim} Ratings by Gender", fontsize=16)
    plt.xlabel(f"{dim} Rating", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.xticks(range(1, 6))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_distributions(female_raw_qoe_df, male_raw_qoe_df, 'QOE')
plot_distributions(female_raw_noi_df, male_raw_noi_df, 'NOI')
plot_distributions(female_raw_col_df, male_raw_col_df, 'COL')
plot_distributions(female_raw_dis_df, male_raw_dis_df, 'DIS')
plot_distributions(female_raw_lou_df, male_raw_lou_df, 'LOU')

def plot_scatter_conditions(f_df, m_df, dim):
    # Extract conditions (c01 to c58) and their corresponding mean ratings
    f_conditions = f_df['File']
    m_conditions = m_df['File']
    f_means = f_df.drop(columns='File').mean(axis=1)
    m_means = m_df.drop(columns='File').mean(axis=1)

    plt.figure(figsize=(10, 6))
    plt.scatter(f_means, m_means, alpha=0.7, color='blue', edgecolor='black')

    f_sems = f_df.drop(columns='File').std(axis=1) / np.sqrt(48)
    m_sems = m_df.drop(columns='File').std(axis=1) / np.sqrt(48)

    plt.errorbar(f_means, m_means, 
                xerr=f_sems, yerr=m_sems, 
                fmt='o', color='blue', alpha=0.6, ecolor='gray', capsize=2, label='Condition Mean ± Standard Error')

    # Annotate points with condition labels
    for f_mean, m_mean, condition in zip(f_means, m_means, f_conditions):
        plt.text(f_mean, m_mean, condition, fontsize=8, ha='right', va='bottom')

    plt.title(f"{dim} Ratings by Condition (Female vs Male)", fontsize=16)
    plt.xlabel(f"Female {dim} Mean Rating", fontsize=14)
    plt.ylabel(f"Male {dim} Mean Rating", fontsize=14)
    plt.axline((1,1), slope=1, color='red', linestyle='--', label='y=x')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_scatter_conditions(female_raw_qoe_df, male_raw_qoe_df, 'QOE')
plot_scatter_conditions(female_raw_noi_df, male_raw_noi_df, 'NOI')
plot_scatter_conditions(female_raw_col_df, male_raw_col_df, 'COL')
plot_scatter_conditions(female_raw_dis_df, male_raw_dis_df, 'DIS')
plot_scatter_conditions(female_raw_lou_df, male_raw_lou_df, 'LOU')

# %%
# Analyse the data across all dimensions according to the ETSI analysis

def calculate_mos_std(df):
    mos = df.drop(columns='File').mean(axis=1)
    std = df.drop(columns='File').std(axis=1)
    return mos, std

def calculate_dmos_std(df, std):
    dmos = df.iloc[0] - df
    new_std = np.sqrt((std.iloc[0] * std.iloc[0]) + (std * std))
    return dmos, new_std

def calculate_t_stat(diff, std_m, std_f, n, df):
    t_stat = diff / np.sqrt((std_m**2 + std_f**2) / n) # Compute t-statistic
    p_value = 2 * t.sf(t_stat, df) # Two-tailed p-value
    return t_stat, p_value

def analyse(male_raw_df, female_raw_df):
    df = pd.DataFrame()
    df['mos_m_CuTi'], df['std_m_CuTi'] = calculate_mos_std(male_raw_df)
    df['mos_f_CuTi'], df['std_f_CuTi'] = calculate_mos_std(female_raw_df)

    df['dcmos'] = df['mos_m_CuTi'] - df['mos_f_CuTi']

    ref_t_stat, ref_p_value = calculate_t_stat(
        abs(df['dcmos'].iloc[0]),
        df['std_m_CuTi'].iloc[0],
        df['std_f_CuTi'].iloc[0],
        48,
        48 - 1)

    df['ref_t_test'] = [ref_p_value] + [np.nan] * (len(df) - 1)

    df['dmos_m_CuTi'], df['new_std_m_CuTi'] = calculate_dmos_std(df['mos_m_CuTi'], df['std_m_CuTi'])
    df['dmos_f_CuTi'], df['new_std_f_CuTi'] = calculate_dmos_std(df['mos_f_CuTi'], df['std_f_CuTi'])

    df['deltadmos'] = df['dmos_m_CuTi'] - df['dmos_f_CuTi']

    con_t_stat, con_p_value = calculate_t_stat(
        abs(df['deltadmos']),
        df['new_std_m_CuTi'],
        df['new_std_f_CuTi'],
        96,
        96 - 1)

    df['con_t_test'] = [f"{x:.8f}" for x in con_p_value]
    df.loc[0, 'con_t_test'] = np.nan

    return df

analysis_qoe = analyse(male_raw_qoe_df, female_raw_qoe_df)
analysis_noi = analyse(male_raw_noi_df, female_raw_noi_df)
analysis_col = analyse(male_raw_col_df, female_raw_col_df)
analysis_dis = analyse(male_raw_dis_df, female_raw_dis_df)
analysis_lou = analyse(male_raw_lou_df, female_raw_lou_df)

analysis_dis
# %%
# Conclusions according to the ETSI analysis
'''def conclusions(df, dim):
    mean = df['deltadmos'].iloc[1:].mean()
    std = df['deltadmos'].iloc[1:].std()
    no_of_conditions = len(df) - 1
    t_stat = abs(mean) / np.sqrt(std / no_of_conditions)
    p_value = 2 * t.sf(t_stat, no_of_conditions - 1)
    print(f"Conclusions for {dim}:")
    print(f"Mean: {mean:.8f}, Std: {std:.8f}, No. of Conditions: {no_of_conditions}")
    print(f"T-statistic: {t_stat:.8f}, P-value: {p_value:.8f}")'''

def conclusions(df, dim):
    # Extract deltaDMOS values excluding the reference condition (row 0)
    delta = df['deltadmos'].iloc[1:]
    
    # Basic stats
    mean = delta.mean()
    std = delta.std()
    n = len(delta)
    
    # T-test (one-sample, H0: mean == 0)
    t_stat = abs(mean) / np.sqrt(std**2 / n)
    p_ttest = 2 * t.sf(t_stat, df=n - 1)
    
    # Shapiro-Wilk normality test
    W_stat, p_shapiro = shapiro(delta)
    
    # Wilcoxon Signed-Rank test (non-parametric)
    w_stat, p_wilcoxon = wilcoxon(delta, alternative='two-sided')

    # Printout
    print(f"Conclusions for {dim}:")
    print(f"Mean: {mean:.8f}, Std: {std:.8f}, N: {n}")
    print(f"Shapiro-Wilk W = {W_stat:.3f}, p = {p_shapiro:.5f} (Normality {'OK' if p_shapiro > 0.05 else 'Violated'})")
    
    if p_shapiro > 0.05:
        print(f"T-test:     T = {t_stat:.5f}, p = {p_ttest:.8f}")
    else:
        print(f"Wilcoxon:   Z = {w_stat:.5f}, p = {p_wilcoxon:.8f}")
    
    # Interpretation suggestion
    if (p_shapiro > 0.05 and p_ttest >= 0.05) or (p_shapiro <= 0.05 and p_wilcoxon >= 0.05):
        print("→ No significant gender-related difference detected.\n")
    else:
        print("→ Significant gender-related difference detected.\n")

conclusions(analysis_qoe, 'QOE')
conclusions(analysis_noi, 'NOI')
conclusions(analysis_col, 'COL')
conclusions(analysis_dis, 'DIS')
conclusions(analysis_lou, 'LOU')

# %%
# Plot scatter plot of all conditions from c02 to c58 for corecon_t_test values under 0.1

# Step 1: Create full condition order
x_order = [f"c{str(i).zfill(2)}" for i in range(2, 59)]
condition_pos = {cond: idx for idx, cond in enumerate(x_order)}

# Step 2: Gather and organize all significant points
all_points = []

for analysis, label in zip(
    [analysis_qoe, analysis_noi, analysis_col, analysis_dis, analysis_lou],
    ['QOE', 'NOI', 'COL', 'DIS', 'LOU']
):
    values = analysis['con_t_test'].iloc[1:].astype(float).round(4)
    for cond, val in zip(x_order, values):
        if val < 0.05:
            all_points.append((condition_pos[cond], val, label))

# Step 3: Plot using x as integer positions
plt.figure(figsize=(14, 8))

for label in ['QOE', 'NOI', 'COL', 'DIS', 'LOU']:
    xs = [x for x, y, l in all_points if l == label]
    ys = [y for x, y, l in all_points if l == label]
    plt.scatter(xs, ys, label=label)

# Step 4: Set fixed tick positions and labels
plt.xticks(ticks=range(len(x_order)), labels=x_order, rotation=90, fontsize=9)
plt.title("Conditions with p-value < 0.05 by Dimension", fontsize=16)
plt.xlabel("Conditions (c02 to c58)", fontsize=14)
plt.ylabel("T-test p-value", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
