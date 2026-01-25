import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_purchase_matrix_and_rank(file_path):
    print("\n--- A1: Purchase Data Analysis ---")

    df = pd.read_excel(file_path, sheet_name="Purchase data")

    X = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
    y = df['Payment (Rs)'].values

    print("Feature Matrix X:\n", X)
    print("Output Vector y:\n", y)

    print("Dimensionality of vector space:", X.shape[1])
    print("Number of vectors:", X.shape[0])

    rank = np.linalg.matrix_rank(X)
    print("Rank of feature matrix:", rank)

    X_pinv = np.linalg.pinv(X)
    cost = X_pinv.dot(y)

    print("Cost of items [Candies, Mangoes, Milk]:")
    print(cost)


def classify_customers_rich_poor(file_path):
    print("\n--- A2: Rich / Poor Classification ---")

    df = pd.read_excel(file_path, sheet_name="Purchase data")

    df['Class'] = df['Payment (Rs)'].apply(lambda x: "RICH" if x > 200 else "POOR")

    print(df[['Payment (Rs)', 'Class']])
    print("Rule: Payment > 200 → RICH else POOR")


def analyze_irctc_stock(file_path):
    print("\n--- A3: IRCTC Stock Analysis ---")

    df = pd.read_excel(file_path, sheet_name="IRCTC Stock Price")

    price = df['Price'].values

    print("Mean Price:", np.mean(price))
    print("Variance:", np.var(price))

    def custom_mean(arr):
        total = 0
        for x in arr:
            total += x
        return total / len(arr)

    def custom_variance(arr):
        m = custom_mean(arr)
        total = 0
        for x in arr:
            total += (x - m) ** 2
        return total / len(arr)

    print("Custom Mean:", custom_mean(price))
    print("Custom Variance:", custom_variance(price))

    wed_data = df[df['Day'] == 'Wed']
    print("Wednesday Mean:", wed_data['Price'].mean())

    april_data = df[df['Month'] == 'Apr']
    print("April Mean:", april_data['Price'].mean())

    loss_prob = len(df[df['Chg%'] < 0]) / len(df)
    print("Probability of Loss:", loss_prob)

    wed_profit = wed_data[wed_data['Chg%'] > 0]
    print("Profit on Wednesday Probability:", len(wed_profit) / len(df))

    print("Conditional P(Profit | Wednesday):", len(wed_profit) / len(wed_data))

    plt.scatter(df['Day'], df['Chg%'])
    plt.title("Change % vs Day")
    plt.xlabel("Day")
    plt.ylabel("Change %")
    plt.show()


def explore_thyroid_dataset(file_path):
    print("\n--- A4: Thyroid Data Exploration ---")

    df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

    print("Data Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nSummary Statistics:\n", df.describe())


def calculate_jaccard_and_smc(file_path):
    print("\n--- A5: Jaccard and SMC ---")

    df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

    numeric_df = df.select_dtypes(include=[np.number]).fillna(0)

    
    binary_df = (numeric_df != 0).astype(int)

    v1 = binary_df.iloc[0].values
    v2 = binary_df.iloc[1].values

    f11 = np.sum((v1 == 1) & (v2 == 1))
    f10 = np.sum((v1 == 1) & (v2 == 0))
    f01 = np.sum((v1 == 0) & (v2 == 1))
    f00 = np.sum((v1 == 0) & (v2 == 0))

    den_jc = (f01 + f10 + f11)
    if den_jc == 0:
        jc = 1.0   
    else:
        jc = f11 / den_jc

    den_smc = (f00 + f01 + f10 + f11)
    if den_smc == 0:
        smc = 0.0
    else:
        smc = (f11 + f00) / den_smc

    print("f00 =", f00, "f01 =", f01, "f10 =", f10, "f11 =", f11)
    print("Jaccard Coefficient:", jc)
    print("Simple Matching Coefficient:", smc)


def calculate_cosine_similarity(file_path):
    print("\n--- A6: Cosine Similarity ---")

    df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")
    data = df.select_dtypes(include=[np.number]).fillna(0)

    v1 = data.iloc[0].values
    v2 = data.iloc[1].values

    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        cos_sim = 0.0
    else:
        cos_sim = np.dot(v1, v2) / denom

    print("Cosine Similarity:", cos_sim)


def plot_similarity_heatmap(file_path):
    print("\n--- A7: Similarity Heatmap ---")

    df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")
    data = df.select_dtypes(include=[np.number]).fillna(0).iloc[:20]

    cosine_matrix = np.zeros((20, 20))

    for i in range(20):
        for j in range(20):
            v1 = data.iloc[i].values
            v2 = data.iloc[j].values
            denom = np.linalg.norm(v1) * np.linalg.norm(v2)

            if denom == 0:
                cosine_matrix[i][j] = 0.0
            else:
                cosine_matrix[i][j] = np.dot(v1, v2) / denom

    sns.heatmap(cosine_matrix, annot=False)
    plt.title("Cosine Similarity Heatmap")
    plt.show()


def main():
    file_path = "Lab2 Session Data.xlsx"

    analyze_purchase_matrix_and_rank(file_path)
    classify_customers_rich_poor(file_path)
    analyze_irctc_stock(file_path)
    explore_thyroid_dataset(file_path)
    calculate_jaccard_and_smc(file_path)
    calculate_cosine_similarity(file_path)
    plot_similarity_heatmap(file_path)


if __name__ == "__main__":
    main()
