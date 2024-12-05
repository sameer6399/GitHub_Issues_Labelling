import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from mlxtend.frequent_patterns import apriori, association_rules
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re

# Ensure required resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Step 1: GitHub API Integration
def fetch_github_issues(repo_owner, repo_name, token, max_issues=100):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues"
    headers = {"Authorization": f"token {token}"}
    params = {"state": "all", "per_page": min(max_issues, 100)}
    issues = []
    
    while url and len(issues) < max_issues:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"Error: {response.status_code}, {response.json()}")
            break
        
        json_response = response.json()
        if not json_response:
            print("No issues found.")
            break
        
        issues.extend(json_response)
        # Pagination
        url = response.links.get('next', {}).get('url', None)
    
    if not issues:
        print("No issues retrieved from the repository.")
    return pd.DataFrame([{
        "title": issue.get("title", ""),
        "body": issue.get("body", ""),
        "labels": [label['name'] for label in issue.get("labels", [])]
    } for issue in issues])


# Step 2: Data Preprocessing
def preprocess_text(text):
    text = re.sub(r'\W+', ' ', text.lower())  # Remove non-alphanumeric characters
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]

# Step 3: Frequent Pattern Mining
def frequent_pattern_mining(texts, min_support=0.1):
    vectorizer = TfidfVectorizer(max_features=100)
    tfidf_matrix = vectorizer.fit_transform([" ".join(text) for text in texts])
    df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    frequent_items = apriori(df > 0, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_items, metric="lift", min_threshold=1.0)
    return frequent_items, rules

# Main Execution
if __name__ == "__main__":
    # Replace these with your repo details and GitHub token
    REPO_OWNER = "RepoOwner"
    REPO_NAME = "RepoName"
    GITHUB_TOKEN = "Token"

    # Fetch issues
    issues_df = fetch_github_issues(REPO_OWNER, REPO_NAME, GITHUB_TOKEN)
    print(f"Fetched {len(issues_df)} issues.")
    print(issues_df.head())  # Inspect the fetched data

    if not issues_df.empty and "body" in issues_df.columns:
        # Preprocess text
        issues_df["processed_body"] = issues_df["body"].fillna("").apply(preprocess_text)
        print("Text preprocessing complete.")

        # Frequent pattern mining
        frequent_items, rules = frequent_pattern_mining(issues_df["processed_body"])
        print("Frequent Patterns:")
        print(frequent_items)
        print("\nAssociation Rules:")
        print(rules)
    else:
        print("No valid issues to process.")
