import requests
import time
import re
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import ndcg_score, precision_score, recall_score, accuracy_score, f1_score

GITHUB_TOKEN = "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"      # NOTE: User may need to insert their own github bearer token if program fails to grab repository data
headers = {"Authorization": f"token {GITHUB_TOKEN}"}
MAX_RETRIES = 3


def load_data():
    print("Loading data...")
    repo_url = "https://api.github.com/repos/calcom/cal.com/commits"
    page_counter = 0
    max_pages = 50  # Set the maximum number of pages you want to fetch

    all_commits = []

    while repo_url and page_counter < max_pages:
        print(f"Searching - Page {page_counter + 1}")
        retries = 0
        success = False
        while retries < MAX_RETRIES and not success:
            response = requests.get(repo_url, headers=headers)

            if response.status_code == 200:
                commits = response.json()

                for commit in commits:
                    sha = commit["sha"]
                    commit_url = f"https://api.github.com/repos/calcom/cal.com/commits/{sha}"
                    detailed_commit = requests.get(commit_url, headers=headers).json()

                    # Extract desired information from detailed_commit
                    modified_files = []
                    for file in detailed_commit.get("files", []):
                        file_data = {
                            "file": file["filename"],
                            "status": file["status"],
                            "additions": file["additions"],
                            "deletions": file["deletions"]
                        }
                        modified_files.append(file_data)

                    commit_data = {
                        "sha": detailed_commit["sha"],
                        "message": detailed_commit["commit"]["message"],
                        "author": detailed_commit["commit"]["author"]["name"],
                        "date": detailed_commit["commit"]["author"]["date"],
                        "modified_files": modified_files
                    }

                    all_commits.append(commit_data)

                # Handle Pagination
                if "next" in response.links:
                    repo_url = response.links["next"]["url"]
                    page_counter += 1  # Increment the page counter
                else:
                    repo_url = None
                success = True

            elif response.status_code == 429:
                # Rate limited; sleeping for the required duration
                retry_after = int(response.headers.get("Retry-After", 60))
                print(f"Rate limited. Sleeping for {retry_after} seconds...")
                time.sleep(retry_after)
                retries += 1

            else:
                print(f"Error {response.status_code}: {response.text}. Retrying...")
                time.sleep(5)
                retries += 1

        if not success:
            print(f"Failed to fetch data after {MAX_RETRIES} retries. Exiting.")
            break

    df = pd.DataFrame(all_commits)
    print(f"Data loading completed - {len(all_commits)} commits")
    return df


def extract_file_ext(filename):
    return filename.split('.')[-1]


def flatten_data(df):
    # Flatten the nested structure
    rows = []
    for index, row in df.iterrows():
        for file in row['modified_files']:
            file_data = file
            file_data['sha'] = row['sha']
            file_data['message'] = row['message']
            file_data['author'] = row['author']
            file_data['date'] = row['date']
            rows.append(file_data)
    return pd.DataFrame(rows)


def handle_missing_data(df):
    print("Handling missing data...")
    df.dropna(inplace=True)
    # Check if there are incomplete records (missing commit messages or file changes) rather than dropping whole records
    return df


def normalize_text_data(df):
    print("Normalizing text data...")
    df["message"] = df["message"].str.lower().str.strip()
    return df


def detect_and_handle_outliers(df):
    print("Detecting and handling outliers...")
    Q1 = (df["additions"]+df["deletions"]).quantile(0.25)
    Q3 = (df["additions"]+df["deletions"]).quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = (Q1 - 1.5) * IQR
    upper_bound = (Q3 + 1.5) * IQR

    # Filter out rows where 'changes' are outside the interquartile range
    df = df[((df["additions"]+df["deletions"]) >= lower_bound) & ((df["additions"]+df["deletions"]) <= upper_bound)]

    return df


def filter_relevant_files(df):
    print("Filtering relevant files...")

    def is_relevant_file(filename):
        irrelevant_extensions = ['.md', '.yml', '.png', '.jpg', '.svg', '.example', 'prisma', 'lock', 'Dockerfile', 'issues', 'conflict']       # Possibly add .json and .sql
        return not any(filename.endswith(ext) for ext in irrelevant_extensions)

    # Apply the filter to each row
    filtered_df = df[df['file'].apply(is_relevant_file)]

    return filtered_df


def remove_duplicates(df):
    print("Removing duplicate data...")
    df.drop_duplicates(inplace=True)
    return df


def create_bug_fix_feature(df):
    print("Creating text analysis feature...")
    bug_related_keywords = ['bug', 'fix', 'error', 'patch', 'issue']
    bug_related_pattern = re.compile('|'.join(bug_related_keywords), re.IGNORECASE)

    def is_bug_related(message):
        return bug_related_pattern.search(message) is not None

    # Apply the function to the 'message' column
    df['is_bug_related'] = df['message'].apply(is_bug_related)

    return df


def create_change_magnitude_feature(df):
    print("Creating change magnitude feature...")
    df['lines_changed'] = df['additions'] + df['deletions']
    return df


def create_commit_frequency_feature(df):
    print("Creating commit frequency feature...")
    commit_frequencies = df.groupby('file')['sha'].count().reset_index(name='commit_frequency')
    df = df.merge(commit_frequencies, on='file')
    return df


def create_commit_time_feature(df):
    print("Creating commit time feature...")
    df['commit_hour'] = pd.to_datetime(df['date']).dt.hour
    # Further categorization into time slots can be added if needed
    return df


def encode_categorical_data(df):
    print("Encoding categorical data...")

    # One-hot encoding for 'file status' and 'file type'
    df = pd.get_dummies(df, columns=['status', 'file_extension'], dtype=int)

    # Binary encoding for 'is_bug_related'
    df['is_bug_related'] = df['is_bug_related'].astype(int)
    return df


def encode_numerical_data(df):
    print("Encoding numerical data...")
    # Standardization
    scaler = StandardScaler()
    df['lines_changed'] = scaler.fit_transform(df[['lines_changed']])
    df['commit_frequency'] = scaler.fit_transform(df[['commit_frequency']])

    # Normalization
    min_max_scaler = MinMaxScaler()
    df['commit_hour'] = min_max_scaler.fit_transform(df[['commit_hour']])
    return df


def clean_data(df):
    df = handle_missing_data(df)
    df = normalize_text_data(df)
    df = filter_relevant_files(df)
    df = detect_and_handle_outliers(df)
    df = remove_duplicates(df)
    return df


def engineer_features(df):
    df = create_bug_fix_feature(df)
    df = create_change_magnitude_feature(df)
    df = create_commit_frequency_feature(df)
    df = create_commit_time_feature(df)
    return df


def preprocess_data(df):
    df = encode_categorical_data(df)
    df = encode_numerical_data(df)
    return df


def prepare_and_group_data(df):
    grouped = df.groupby('sha')
    X = df.drop(['additions', 'deletions', 'sha', 'message', 'author', 'date',  'is_bug_related'], axis=1)
    y = df['is_bug_related']    # Relevance label
    groups = grouped.size().to_list()
    return X, y, groups


def create_lgb_dataset(X, y, groups):
    return lgb.Dataset(X, label=y, group=groups)


def train(train_data, feature_names):
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'learning_rate': 0.1,
        'num_leaves': 31,
        'verbose': 1
    }
    gbm = lgb.train(params, train_data, num_boost_round=100, feature_name=feature_names)
    return gbm


def evaluate(model, X_features, y, threshold=0.5):
    y_pred = model.predict(X_features)

    ndcg = ndcg_score([y], [y_pred])
    print(f"NDCG Score: {ndcg}")

    # Classify files as 'buggy' or 'not buggy' based on the threshold
    y_pred_classified = (y_pred >= threshold).astype(int)
    return y_pred, y_pred_classified


def calculate_metrics(y_true, y_pred_classified):
    precision = precision_score(y_true, y_pred_classified)
    recall = recall_score(y_true, y_pred_classified)
    accuracy = accuracy_score(y_true, y_pred_classified)
    f1 = f1_score(y_true, y_pred_classified)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")


def create_ranked_df(X, y_pred):
    # Add the predictions to X
    X['relevance_score'] = y_pred

    # Sort the files by relevance score in descending order
    ranked_df = X.sort_values(by='relevance_score', ascending=False)
    return ranked_df[['file', 'relevance_score']]


def visualize_top_ranked_files(ranked_df):
    relevance_scores = ranked_df['relevance_score']

    plt.figure(figsize=(10, 6))
    sns.histplot(relevance_scores, kde=True, bins=30, color='skyblue')
    plt.title('Relevance Score Distribution')
    plt.xlabel('Relevance Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def main():
    df = load_data()
    flattened_df = flatten_data(df)
    flattened_df = clean_data(flattened_df)
    flattened_df['file_extension'] = flattened_df['file'].apply(extract_file_ext)
    flattened_df = engineer_features(flattened_df)
    flattened_df = preprocess_data(flattened_df)

    # Preparing data for L2R
    X, y, groups = prepare_and_group_data(flattened_df)
    feature_names = [col for col in X.columns if col != 'file']     # Exclude 'file' from features
    X_features = X[feature_names]
    train_data = create_lgb_dataset(X_features, y, groups)

    # Training the model
    lgb_model = train(train_data, feature_names)

    # Evaluating the model and get predictions
    y_pred, y_pred_classified = evaluate(lgb_model, X_features, y)
    calculate_metrics(y, y_pred_classified)

    # Get predictions and list ranked files and their relevance scores
    ranked_df = create_ranked_df(X, y_pred)
    ranked_df.to_csv('ranked_files.csv', index=False)

    # Save processed data
    flattened_df.to_csv('processed_dataframe.csv', index=False)

    visualize_top_ranked_files(ranked_df)


if __name__ == "__main__":
    main()
