# ${Sentiment140}$
Kaggle dataset: https://www.kaggle.com/datasets/kazanova/sentiment140
Used: Tensorflow, PyTorch, PySpark, Pandas, nltk
To be updated: Use transformers and bert. Config variables will be parsed from cmd. Adjust directories.
Worth saying, data preprocessing using pandas took approximitaly 400 seconds, while same processing using PySpark took approximatly 120 seconds. But the reason for using pandas over PySpark that PySpark cleaning resulted in some Nulls unlike Pandas, it didn't affect anyway but worth saying.
Sentiment140/
├── README.md             # overview of the project
├── data/                 # data files used in the project
│   ├── README.md         # describes where data came from
│   └── sub-folder/       # may contain subdirectories
├── processed_data/       # intermediate files from the analysis
├── manuscript/           # manuscript describing the results
├── results/              # results of the analysis (data, tables, figures)
├── src/                  # contains all code in the project
│   ├── LICENSE           # license for your code
│   ├── requirements.txt  # software requirements and dependencies
│   └── ...
└── doc/                  # documentation for your project
    ├── index.rst
    └── ...
