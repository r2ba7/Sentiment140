# ${Sentiment140}$
## ${Dataset Link}$
* Kaggle dataset: https://www.kaggle.com/datasets/kazanova/sentiment140
## ${Used Frameworks/Libraries}$
* Tensorflow, PyTorch, PyTorch Lightining, transformers, PySpark, Pandas, nltk
## ${Preprocessing notes}$
* Worth saying, data preprocessing using pandas took approximitaly 400 seconds, while same processing using PySpark took approximatly 120 seconds. But the reason for using pandas over PySpark that PySpark cleaning resulted in some Nulls unlike Pandas, it didn't affect anyway but worth saying.
## ${Project \space Structure}$
    ├── README.md          
    ├── original           
    ├── src                <- Source code folder for this project
        │
        ├── data           <- Datasets used and collected for this project
        │   
        ├── docs           <- Folder for Task documentations, Meeting Presentations and task Workflow Documents and Diagrams.
        │
        ├── references     <- Data dictionaries, manuals, and all other explanatory references used 
        │
        ├── tasks          <- Master folder for all individual task folders
        │
        ├── visualizations <- Code and Visualization dashboards generated for the project
        │
        └── results        <- Folder to store Final analysis and modelling results and code.
--------
