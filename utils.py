from imports import *

class Config:
    TRAIN_SIZE = 0.8
    MAX_NB_WORDS = 100000
    MAX_SEQUENCE_LENGTH = 30
    PATH = "preprocessed.csv"
    NUM_EPOCHS = 3
    HIDDEN_SIZE = 64
    OUTPUT_SIZE = 1
    BATCH_SIZE = 256    
    LR = 0.01
    LSTM_LAYERS = 1
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def METRIC(outputs, labels):
        """
        Binary classification Accuracy Metric.
        """
        outputs = outputs > 0.5
        return (labels == outputs).sum().item() / labels.size(0)

class Text_Preprocessing:
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
        STOP_WORDS = stopwords.words('english')
        return ' '.join([word for word in text.split() if word not in STOP_WORDS])

    def stemming(text: str) -> str:
        """
        Stems the text.
        """
        stemmer = SnowballStemmer("english")
        return ' '.join([stemmer.stem(word) for word in text.split()])

    def lemmatization(text: str) -> str:
        """
        Lemmatization is the process of grouping together the inflected forms of a word.
        Parameters:
            text: str
        """
        lemmatizer = WordNetLemmatizer()
        return ' '.join([lemmatizer.lemmatize(word, 'v') for word in text.split()])

    def tokenize(df: DataFrame, i_p: StringType) -> DataFrame:
        """
        Tokenizes the text.
        Parameters:
            df: DataFrame
        """
        token = Tokenizer(inputCol=i_p, outputCol="words")
        return token.transform(df)

    def preprocess_text(text: str, stem=True) -> str:
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