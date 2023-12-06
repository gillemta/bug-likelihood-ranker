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
1. modify the 'max_pages' variable to the amount of pages of commit data you wish to pull in (The default is 50 pages)
```max_pages = 50```

Run the main program:
```python main.py```
