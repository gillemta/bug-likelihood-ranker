# Bug Prediction Using Learning to Rank (L2R)

A Python tool using machine learning to rank repository files by bug likelihood. It integrates Learning-to-Rank techniques for efficient bug prediction and code quality enhancement.

## Features
- Data extraction from GitHub using the GitHub API
- Comprehensive data cleaning and preprocessing
- Feature engineering to extract meaningful attributes from commit data
- Implementation of L2R using the LightGBM framework
- Evaluation of the model using NDCG score, precision, recall, accuracy, and F1 score

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes

### Prerequisites
- Python 3.x
- Pandas
- LightGBM
- Matplotlib
- Seaborn
- Scikit-learn

### Installing
1. Clone the repository:
```git clone https://github.com/<yourusername>/bug-likelihood-ranker.git```
2. Navigate to the project directory and install the required packages:
```pip install -r requirements.txt```

### Usage
1. Insert Bearer Token so you can use the GitHub API
```GITHUB_TOKEN = "<insert-token-here>"```
2. modify the 'max_pages' variable to the amount of pages of commit data you wish to pull in (The default is 50 pages)
```max_pages = 50```

3. Run the main program:
```python main.py```

### Bearer Token
1. **Log in to GitHub**: Go to GitHub and sign in to your account
2. **Access Token Settings**: Click on you profile picture in the top-right corner, then select `Settings`
3. **Developer Settings**: On the settings page, scroll down to the `Developer Settings` section and click on it
4. **Personal Access Token**: In the developer settings, select `Personal access tokens`
5. **Generate New Token**: Click the `Generate new token` button
6. **Token Description**: Enter a descriptive name for the token in the `Note` field, such as "My Project Token"
7. **Select Scopes**: Choose the permissions (scopes) required for the token. Form most projects, the `repo` scope suffices.
8. **Generate Token**: Click the `Generate token` button at the bottom of the page
9. **Copy the Token**: Copy the token immediately, as you wont be able to see it again after leaving the page
10. **Paste Token in Program**: Insert Bearer Token so you can use the GitHub API
```GITHUB_TOKEN = "<insert-token-here>"```

#### Errors
If when running you get an error...
```Error 401: {"message":"Bad credentials","documentation_url":"https://docs.github.com/rest"}```
You have set up the bearer token incorrectly.


#### Documentation on how to setup a bearer token in github can be found at:
https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
