# Sentiment140
Kaggle dataset: https://www.kaggle.com/datasets/kazanova/sentiment140
Used: Tensorflow, PyTorch, PySpark, Pandas, nltk
To be updated: Use transformers and bert. Config variables will be parsed from cmd. Adjust directories.
Worth saying, data preprocessing using pandas took approximitaly 400 seconds, while same processing using PySpark took approximatly 120 seconds. But the reason for using pandas over PySpark that PySpark cleaning resulted in some Nulls unlike Pandas, it didn't affect anyway but worth saying.
