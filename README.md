# ${Sentiment140}$
## ${Dataset \space Link}$
* Kaggle dataset: https://www.kaggle.com/datasets/kazanova/sentiment140
## ${Used Frameworks/Libraries}$
* Tensorflow, PyTorch, PyTorch Lightining, transformers, PySpark, Pandas, nltk
## ${Preprocessing \space notes}$
* Worth saying, data preprocessing using pandas took approximitaly 400 seconds, while same processing using PySpark took approximatly 120 seconds. But the reason for using pandas over PySpark that PySpark cleaning resulted in some Nulls unlike Pandas, it didn't affect anyway but worth saying.
## ${Project \space Structure}$
    ├── README.md          
    ├── original           
    ├── src
        │
        ├── data         
        │   
        ├── notebooks      
        │
        └── python files
--------
