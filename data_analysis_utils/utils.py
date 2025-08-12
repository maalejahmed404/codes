
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
from IPython.display import display
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import kendalltau

cm = sns.light_palette("red", as_cmap=True)

def check_duplicates_report(df):
    duplicates_count = df.duplicated().sum()
    total_rows = len(df)
    
    print("=" * 40)
    
    if duplicates_count == 0:
        print(f" No duplicates found in {total_rows:,} rows")
    else:
        print(f"  {duplicates_count} duplicates found ({duplicates_count/total_rows:.2%})")
        print(f"    Total rows affected: {duplicates_count:,}/{total_rows:,}")

def detect_peaks_in_numeric(
    df,
    feature: str,
    bins: int = 100,
    sigma: float = 2,
    peak_distance: int = 5,
    peak_height: float = 0.001,
    exclude_values: list = None,
    show_plot: bool = True
):

    data = df[feature].dropna()

    if exclude_values is not None:
        data = data[~data.isin(exclude_values)]

    hist_vals, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    smoothed = gaussian_filter1d(hist_vals, sigma=sigma)

    peaks, _ = find_peaks(smoothed, distance=peak_distance, height=peak_height)
    peak_positions = bin_centers[peaks]

    if show_plot:
        plt.figure(figsize=(8, 4))
        plt.plot(bin_centers, smoothed, label='Smoothed Density')
        plt.plot(peak_positions, smoothed[peaks], 'ro', label='Detected Peaks')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.title(f'Peak Detection on {feature}')
        plt.legend()
        plt.show()

    return peak_positions, bin_edges

def checking_outlier(list_feature, df):
    print("=" * 40)
    outlier_info = []
    
    for feature in list_feature:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)][feature]
        outlier_info.append({
            "Feature": feature,
            "Count": len(outliers),
            "Percentage": round(len(outliers) / len(df) * 100, 2),
            "Lower Bound": lower_bound,
            "Upper Bound": upper_bound,
            "Min Outlier": outliers.min() if len(outliers) > 0 else None,
            "Max Outlier": outliers.max() if len(outliers) > 0 else None,
            "Mean Outlier": outliers.mean() if len(outliers) > 0 else None,
        })
    
    return pd.DataFrame(outlier_info)

def color(n_colors=2):
    cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
    positions = np.linspace(0, 1, n_colors)
    colors = [cmap(p) for p in positions]
    return colors

def plot_numerical_features(df_train,num_features):
    colors = color(n_colors=3)
    n = len(num_features)

    fig, axes = plt.subplots(n, 2, figsize=(12, n * 4))
    axes = np.array(axes).reshape(n, 2)

    for i, feature in enumerate(num_features):
        sns.histplot(data=df_train[feature], color=colors[0], bins=20, kde=True, ax=axes[i, 0], label="Train data")
        axes[i, 0].set_title(f"Histogram of {feature}", pad=14, weight="bold")
        axes[i, 0].legend()
        sns.despine(left=False, bottom=False, ax=axes[i, 0])

        df_plot = pd.DataFrame({"Dataset": "Train data", feature: df_train[feature]}).reset_index(drop=True)
        sns.boxplot(data=df_plot, x=feature, y="Dataset", palette=colors, orient="h", ax=axes[i, 1])
        axes[i, 1].set_title(f"Horizontal Box plot of {feature}", pad=14, weight="bold")
        sns.despine(left=False, bottom=False, ax=axes[i, 1])

    plt.tight_layout()
    plt.show()

def check_skewness(data, numerical_features , highlight=True, sort=True):
    skewness_dict = {}
    skew_feature = []
    for feature in numerical_features:
        skew = data[feature].skew(skipna=True)
        skewness_dict[feature] = skew

    skew_df = pd.DataFrame.from_dict(skewness_dict, orient="index", columns=["Skewness"])
    if sort:
        skew_df = skew_df.reindex(skew_df["Skewness"].abs().sort_values(ascending=False).index)

    print(f"{'Feature':<30} | {'Skewness':<9} | {'Remark'}")
    print("-"*70)
    for feature, row in skew_df.iterrows():
        skew = row["Skewness"]
        abs_skew = abs(skew)
        if abs_skew > 1:
            remark = "Highly skewed"
            color = "\033[91m"  
        elif abs_skew > 0.5:
            remark = "Moderately skewed"
            color = "\033[93m"  
        else:
            remark = "Approximately symmetric"
            color = ""
        endc = "\033[0m" if color else ""
        if highlight and color:
            print(f"{color}{feature:<30} | {skew:>+9.4f} | {remark}{endc}")
            skew_feature.append(feature)
        else:
            print(f"{feature:<30} | {skew:>+9.4f} | {remark}")
    print("-"*70)
    return skew_feature, skew_df

def analyze_categorical(df, cols):
    results = {}
    for col in cols:
        freq = df[col].value_counts(dropna=False)
        results[col] = {
            "count": df[col].count(),
            "unique": df[col].nunique(dropna=False),
            "top": freq.index[0],
            "freq": freq.iloc[0],
            "second_top": freq.index[1] if len(freq) > 1 else None,
            "second_freq": freq.iloc[1] if len(freq) > 1 else None,
            "missing_values": df[col].isna().sum(),
            "most_common_values": freq.to_dict()
        }
    return pd.DataFrame(results).T

def plot_correlation(df_train):
    corr_train = df_train.corr(numeric_only=True)
    mask_train = np.triu(np.ones_like(corr_train, dtype=bool))
    cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)

    fig, ax = plt.subplots(figsize=(24, 10))
    sns.heatmap(data=corr_train,
                mask=mask_train,
                annot=True,
                fmt=".2f",
                cmap=cmap,
                vmin=-1, vmax=1,
                linecolor="white",
                linewidths=0.5,
                ax=ax)
    plt.tight_layout()
    plt.show()
