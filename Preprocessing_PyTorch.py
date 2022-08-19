from imports import *
from utils import *

class Preprocessing:
    """
    This class was created to preprocess the Pandas dataframe
    Instead of PySpark because PySpark dataframe when converted to Pandas
    Dataframe, there was missing values.
    """
    def __init__(self):
        """
        Initializes the preprocessing class.
        """
        self.path = Config.PATH
        self.len = Config.MAX_SEQUENCE_LENGTH
        self.max_words = Config.MAX_NB_WORDS
        self.train_size = Config.TRAIN_SIZE
        self.df = self.load_data()

    def load_data(self):
        """
        Loads the data.
        """
        return pd.read_csv(self.path)


    def preprocesss(self):
        """
        Preprocesses the data.
        """
        self.df['cleaned'] = self.df.text.apply(Text_Preprocessing.preprocess_text)
        self.target = self.df.y
        self.df.drop(['text', 'y', 'cleaned_text'], axis=1, inplace=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df, self.target, 
                                                                                test_size=1-self.train_size, 
                                                                                random_state=42)
        print("Done preprocessing.")

    def prepare_tokens(self):
        """
        Prepares the tokens.
        """
        self.tokenizer = Tokenizer(oov_token='UNK')
        self.tokenizer.fit_on_texts(self.X_train.cleaned)

    def get_tokenizer(self):
        """
        Returns the tokenizer.
        """
        return self.tokenizer
    
    def sequence_to_token(self):
        """
        Converts the text to sequences.
        """
        self.X_train = self.tokenizer.texts_to_sequences(self.X_train.cleaned)
        self.X_test = self.tokenizer.texts_to_sequences(self.X_test.cleaned)
        
    def padding(self):
        """
        Pads the sequences.
        """
        self.X_train = sequence.pad_sequences(self.X_train, maxlen=self.len, padding='post')
        self.X_test = sequence.pad_sequences(self.X_test, maxlen=self.len, padding='post')

    def adjust_outputs(self):
        """
        Adjusts the outputs.
        """
        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)
        self.y_train = self.y_train.reshape(-1, 1)
        self.y_test = self.y_test.reshape(-1, 1)

    def text2seq(self):
        """
        Converts the text to sequences.
        """
        self.preprocesss()
        self.prepare_tokens()
        self.sequence_to_token()
        self.padding()
        self.adjust_outputs()
        print("Done text2seq.")

    def get_data(self):
        """
        Returns the data.
        """
        return self.X_train, self.X_test, self.y_train, self.y_test