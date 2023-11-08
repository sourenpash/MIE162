# Load the dataset
df = pd.read_csv("clean_kaggle_data_2022.csv")

# Handling missing values by filling 'null' with 'unknown'

df = df.drop(["Q5"], axis=1)
df = df.drop(["Duration (in seconds)", "Q29"], axis=1)
df = df.drop(0)


# Set encoding labels manually for ordinal columns
Q2_encode = {
    "18-21": 0,
    "22-24": 1,
    "25-29": 2,
    "30-34": 3,
    "35-39": 4,
    "40-44": 5,
    "45-49": 6,
    "50-54": 7,
    "55-59": 8,
    "60-64": 9,
    "65-69:": 10,
    "70+": 11,
}
Q11_encode = {
    "I have never written code": 0,
    "<1 years": 1,
    "1-3 years": 2,
    "3-5 years": 3,
    "5-10 years": 4,
    "10-20 years": 5,
    "20+ years": 6,
}
Q16_encode = {
    "I do not use machine learning methods": 0,
    "Under 1 year": 1,
    "1-2 years": 2,
    "2-3 years": 3,
    "3-4 years": 4,
    "4-5 years": 5,
    "5-10 years": 6,
    "10-20 years": 7,
}
Q25_encode = {
    "0-49 employees": 0,
    "50-249 employees": 1,
    "250-999 employees": 2,
    "1000-9,999 employees": 3,
    "10,000 or more employees": 4,
}
Q26_encode = {
    "0": 0,
    "2-Jan": 1,
    "4-Mar": 2,
    "9-May": 3,
    "14-Oct": 4,
    "15-19": 5,
    "20+": 6,
}
Q30_encode = {
    "$0 ($USD)": 0,
    "$1-$99": 1,
    "$100-$999": 2,
    "$1000-$9,999": 3,
    "$0 ($USD)": 4,
    "$10,000-$99,999": 5,
    "$100,000 or more ($USD)": 6,
}
Q43_encode = {
    "Never": 0,
    "Once": 1,
    "2-5 times": 2,
    "6-25 times": 3,
    "More than 25 times": 4,
}

# Encode ordinal columns of dataset
df["Q2"] = df["Q2"].map(Q2_encode)
df["Q11"] = df["Q11"].map(Q11_encode)
df["Q16"] = df["Q16"].map(Q16_encode)
df["Q25"] = df["Q25"].map(Q25_encode)
df["Q26"] = df["Q26"].map(Q26_encode)
df["Q30"] = df["Q30"].map(Q30_encode)
df["Q43"] = df["Q43"].map(Q43_encode)

# Fill in missing data in ordinal columns with its mean
col_ordinal = ["Q2", "Q11", "Q16", "Q25", "Q26", "Q30", "Q43"]
for col in col_ordinal:
    df[col].fillna(df[col].mean(), inplace=True)

# Encode Q9 column from Y/N to 0 and 1 and fill null with mean
df["Q9"] = df["Q9"].replace({"No": 0, "Yes": 1})
df["Q9"].fillna(df["Q9"].mean(), inplace=True)

# Temporarily remove ordinal and target columns
removed_col = df[
    ["Q2", "Q11", "Q16", "Q25", "Q26", "Q30", "Q43", "Q9", "Q29_Encoded", "Q29_buckets"]
]
df = df.drop(
    columns=[
        "Q2",
        "Q11",
        "Q16",
        "Q25",
        "Q26",
        "Q30",
        "Q43",
        "Q9",
        "Q29_Encoded",
        "Q29_buckets",
    ]
)

# Do one-hot encoding on rest of dataframe
df_encoded = pd.get_dummies(df)

# Reintroduce removed columns back into the dataframe
df_encoded = pd.concat([df_encoded, removed_col], axis=1)
df_encoded.head()

# Create X (features) and y (target variables) dataframes for feature selection
X = df_encoded.drop(columns=["Q29_Encoded", "Q29_buckets"], axis=1)
y = df_encoded["Q29_Encoded"]

# Calculate chi squared scores of all features
chi_scores = chi2(X, y)
chi_scores
# Set features to be selected through K best
selector = SelectKBest(score_func=chi2, k=k)
X_new = selector.fit_transform(X, y)
selected_feature_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_feature_indices]
selected_features  # shows selected features in alphabetical order
# Filter encoded dataframe to only have selected features
df_selected = df_encoded[selected_features]
df_selected = pd.concat(
    [df_encoded[selected_features], df_encoded[["Q29_Encoded", "Q29_buckets"]]], axis=1
)
df_selected.head()

# Create X and y dataframes for splitting and training
X = df_selected.drop(columns=["Q29_Encoded", "Q29_buckets"], axis=1)
y = df_selected["Q29_Encoded"]

# Split X and y with an 80:20 training-to-testing split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
