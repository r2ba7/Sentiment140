from imports import *
warnings.filterwarnings("ignore")

def load_data(path, schema, CSV=True, provide_header=False):
    """
    Loads the data from the given path.
    """
    if CSV:
        if provide_header:
            df = sc.read.format("csv")\
                .option("header", "false")\
                .schema(schema)\
                .load(path)
        else:
            df = sc.read.csv(path, header=True, inferSchema=True)
    else:
        df = sc.read.json(path)
    df.persist()
    return df

def show_df(df: DataFrame)-> DataFrame:
    """
    Shows the dataframe.
    """
    return df.show()

def describe_data(df):
    """
    Prints the dataframe's statistics.
    """
    return(df.describe().show())

def check_nan(df: DataFrame) -> DataFrame:
    """
    Checks for NaN values in the dataframe.
    """
    return(df.select([count(when(isnan(c), c)).alias(c) for c in df.columns]).show())

def check_nunique(df: DataFrame)-> DataFrame:
    """
    Checks the number of unique values in the dataframe.
    """
    return(df.select([count(when(c.isNotNull(), c)).alias(c) for c in df.columns]).show())

def drop_columns(df: DataFrame, cols: list) -> DataFrame:
    """
    Drops the columns that are not needed.
    """
    return df.drop(*cols)

def map_label(y):
    """
    Maps the label to nicer presentation
    """
    return 0 if y == 0 else 1 if y == 4 else 2

def split(df: DataFrame, train_size: float, valid_size:float, test_size:float) -> DataFrame:
    """
    Splits the dataframe into train and test sets.
    """
    assert (train_size + valid_size + test_size) == 1.0, "The sum of train, valid and test sizes must be 1.0"
    return df.randomSplit([train_size, valid_size, test_size], seed=42)

def Preprocessing(df: DataFrame, stem: bool=True) -> DataFrame:
    """
    Preprocesses the dataframe.
    """
    STOP_WORDS = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    stemmer = SnowballStemmer("english")
    #text = df.select("text")
    def text_cleaning2(text: str) -> str:
        text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
        return re.sub(text_cleaning_re, " ", text.lower()).strip()


    def text_cleaning(text: str) -> str:
        """
        Cleans the text.
        """
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'bit.ly/\S+', '', text)
        text = text.strip('[link]')

        # remove users
        text = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', text)
        text = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', text)

        # remove puntuation
        my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@â'
        text = re.sub('[' + my_punctuation + ']+', ' ', text)

        # remove number
        text = re.sub('([0-9]+)', '', text)

        # remove hashtag
        text = re.sub('(#[A-Za-z]+[A-Za-z0-9-_]+)', '', text)
        return text.lower()
        
    def remove_stopwords(text: str) -> str:
        """
        Removes the stopwords from the text.
        """
        return ' '.join([word for word in text.split() if word not in STOP_WORDS])

    def stemming(text: str) -> str:
        """
        Stems the text.
        """
        return ' '.join([stemmer.stem(word) for word in text.split()])

    def lemmatization(text: str) -> str:
        """
        Lemmatization is the process of grouping together the inflected forms of a word.
        Parameters:
            text: str
        """
        return ' '.join([lemmatizer.lemmatize(word, 'v') for word in text.split()])

    def tokenize(df: DataFrame, i_p: StringType) -> DataFrame:
        """
        Tokenizes the text.
        Parameters:
            df: DataFrame
        """
        token = Tokenizer(inputCol=i_p, outputCol="words")
        return token.transform(df)

    def preprocess_text(text: str) -> str:
        """
        Preprocesses the text.
        Parameters:
            text: the text to preprocess.
            stem: if True, stems the text.
        """
        if stem:
            statement = stemming(remove_stopwords(text_cleaning2(text)))
            return statement
        else:
            statement = lemmatization(remove_stopwords(text_cleaning2(text)))
            return statement

    cleaned_text = udf(lambda x: preprocess_text(x), StringType())
    df = df.withColumn("cleaned_text", cleaned_text("text"))
    return df

if __name__ == "__main__":
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    PATH = "training.1600000.processed.noemoticon.csv"
    print("Starting the program...")
    beginning = time.time()
    sc = SparkSession.builder.master("local[*]").appName("Sentiment Analysis").getOrCreate()
    schema = StructType([\
        StructField("y", IntegerType(), True),\
        StructField("ids", IntegerType(), True),\
        StructField("date", StringType(), True),\
        StructField("flag", StringType(), True),\
        StructField("user", StringType(), True),\
        StructField("text", StringType(), True)])
    y_schema = StructType([\
        StructField("y", IntegerType(), True)])
    DROPPED_COLS = ['ids', 'date', 'flag', 'user']
    df = load_data(PATH, schema, provide_header=True)
    df = drop_columns(df, DROPPED_COLS)
    df = Preprocessing(df, True)
    label = udf(lambda x: map_label(x), IntegerType())
    df = df.withColumn("y", label("y"))
    final_df = df.toPandas()
    final_df.to_csv("wow.csv", index=False)
    print("The program has finished. Preprocessing took {} seconds".format(time.time() - beginning))