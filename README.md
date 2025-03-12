# Bug Prediction Using Learning to Rank (L2R)

A Python tool using machine learning to rank repository files by bug likelihood. It integrates Learning-to-Rank techniques for efficient bug prediction and code quality enhancement.

---

## Features
- **Data Extraction:**  
  Retrieve commit data from GitHub using the GitHub API.
- **Data Preprocessing:**  
  Comprehensive cleaning, normalization, and feature engineering on commit data.
- **Learning-to-Rank Implementation:**  
  Utilizes LightGBM to rank source code files based on bug likelihood.
- **Model Evaluation:**  
  Evaluated using metrics such as NDCG score, precision, recall, accuracy, and F1 score.

---

## Research Artifacts
For an in-depth look at the methodology, experimental evaluation, and outcomes, please refer to the following documents:
- [Research Paper: Ranking Source Code for Bug Prediction: An L2R Approach on An Open Source Repository](Final Project Powerpoint.pdf)
- [Presentation: Bug Prediction Using Learning-to-Rank](Final Project Powerpoint.pptx)

---

## Getting Started

### Prerequisites
- Python 3.x
- Pandas
- LightGBM
- Matplotlib
- Seaborn
- Scikit-learn

### Installing
1. **Clone the Repository:**
  ```bash
  git clone https://github.com/gillemta/bug-likelihood-ranker.git
  ```

2. **Navigate to the Project Directory and Install Dependencies:**
  ```bash
  cd bug-likelihood-ranker
  pip install -r requirements.txt
  ```

---

## Usage

1. **Insert Your GitHub API Bearer Token:**
  In the main script, set your bearer token:
  ```python
  GITHUB_TOKEN = "<insert-token-here>"
  ```

2. **Configure Data Retrieval:**
  Modify the `max_pages` variable to specify the number of pages of commit data to pull (default is 50):
  ```python
  max_pages = 50
  ```

3. **Run the Main Program**:
  ```bash
  python main.py
  ```

### Bearer Token Setup

1. **Log in to GitHub:** Sign in to your GitHub account.
2. **Access Token Settings:** Click your profile picture in the top-right corner, then select `Settings`.
3. **Developer Settings:** Scroll down and click on `Developer Settings`.
4. **Personal Access Tokens:** Select `Personal access tokens`.
5. **Generate a New Token:**
  Click `Generate new token`, provide a descriptive name (e.g., "Bug Likelihood Ranker Token"), and select the required scopes (typically the `repo` scope).
6. **Generate and Copy the Token:**
  Click `Generate token` and immediately copy the token, as it will not be shown again.
7. **Insert the Token:**
  Paste the token into your program where indicated.

## Troubleshooting
If you receive an error such as:
```json
Error 401: {"message":"Bad credentials","documentation_url":"https://docs.github.com/rest"}
```
Double-check your token and ensure it’s correctly inserted. For more detailed guidance, refer to [GitHub's documentation on personal access tokens](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).

---

## Screenshots / Demo

Below is a demo GIF showcasing the ranking results and key outputs:

![Ranking Demo](assets/ranking-demo.gif)

*Replace `assets/ranking-demo.gif` with the actual path or URL to your demo GIF or screenshots.*
