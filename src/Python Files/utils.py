from imports import *

class Config:
    TRAIN_SIZE = 0.7
    MAX_NB_WORDS = 100000
    MAX_SEQUENCE_LENGTH = 150
    PATH = "preprocessed.csv"
    NUM_EPOCHS = 10
    HIDDEN_SIZE = 128
    OUTPUT_SIZE = 1
    BATCH_SIZE = 64    
    LR = 0.001
    LSTM_LAYERS = 5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def METRIC(outputs, labels):
        """
        Binary classification Accuracy Metric.
        """
        return (outputs.argmax(1) == labels).type(torch.float).sum().item()

class TextPreprocessor:

    def __init__(self):
        pass
    
    def decontract(text):
            text = re.sub(r"can\'t", "can not", text)
            text = re.sub(r"n\'t", " not", text)
            text = re.sub(r"\'re", " are", text)
            text = re.sub(r"\'s", " is", text)
            text = re.sub(r"\'d", " would", text)
            text = re.sub(r"\'ll", " will", text)
            text = re.sub(r"\'t", " not", text)
            text = re.sub(r"\'ve", " have", text)
            text = re.sub(r"\'m", " am", text)
            return text
    
    def clean_text(text):
        """
        ChatGPT
        """
        # Remove links
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'www\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\S+', '', text)
        text = re.sub(r'#\S+', '', text)

        # Remove emojis
        text = re.sub(r'[\U0001f600-\U0001f650]', '', text)
        
        # Remove punctuation and convert to lowercase
        text = ''.join([c for c in text if c not in punctuation])
        text = text.lower().strip()
        
        # Remove stop words and tokenize
        # stop_words = set(stopwords.words('english'))
        # tokens = word_tokenize(text)
        # tokens = [token for token in tokens if token not in stop_words]
        
        # # Join the tokens back into a string
        # text = ' '.join(tokens)
        
        return text


    def text_cleaning2(text: str) -> str:
        """
        Cleans the text.
        """
        # text_no_emo = re.sub(r'[\:\;\=]\s*[D\)\(\[\]\}\{@\|\\\/]', '', text) #remove emo text
        cleaned_text = re.sub("@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+", " ", text).strip().lower()
        return cleaned_text

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
        custom_stopwords = set(stopwords.words('english')) - {'not', 'no', 'never'}
        return ' '.join([word for word in text.split() if word not in custom_stopwords])

    def stemming(text: str) -> str:
        """
        Stems the text.
        """
        tokenized = nltk.word_tokenize(text)
        stemmer = PorterStemmer()
        return ' '.join([stemmer.stem(word) for word in tokenized])


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
            else, lemmatizes the text.
        """
        if stem:
            return TextPreprocessor.stemming(TextPreprocessor.remove_stopwords(TextPreprocessor.decontract(TextPreprocessor.clean_text(text))))
        else:
            return TextPreprocessor.lemmatization(TextPreprocessor.remove_stopwords(TextPreprocessor.decontract(TextPreprocessor.clean_text(text))))