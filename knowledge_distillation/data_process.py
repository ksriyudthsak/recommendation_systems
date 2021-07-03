import re
import string
import pandas as pd 
import nltk 

class DataProcessing:
    def __init__(self):
        self.text = "title"
        self.label = "category_flag"
        self.filename_all = "news_click_df_select.csv"
        self.filename = "news_click_df.csv"

    def get_data(self, data_path, text_columns, label_columns):
        filename_ = data_path + self.filename_all
        data_df = pd.read_csv(filename_)
        print (data_df["click_prob_flag"].value_counts())
        print (data_df["category_flag"].value_counts())
        data_df["text"] = data_df[text_columns] #+ ". " + data_df["abstract"]
        data_df["label"] = data_df[label_columns] # category_flag # click_prob_flag
        display (data_df.head(2))
        return data_df

    def select_data(self, data_df, select_col, decrease_fold=1):
        select_data_df = data_df[select_col].tail(int(len(data_df)/int(decrease_fold))) # decrease the number of data
        print (select_data_df["label"].value_counts())
        return select_data_df

    def save_data_to_csv(self, df, data_path, select_columns=["text_clean","label"]):
        select_df = df[select_columns]
        filename_ = data_path + self.filename
        print("filename_", filename_)
        select_df.to_csv(filename_, index=False)
        print (select_df["label"].value_counts())
        return

    def clean_text(self, text):
        # make text lowercase, remove text in square brackets,
        # remove links,remove punctuation and remove words containing numbers.
        text = text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        return text

    def text_preprocessing(self, text):
        # cleaning and parsing the text.
        tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        nopunc = self.clean_text(text)
        tokenized_text = tokenizer.tokenize(nopunc)
        #remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]
        combined_text = ' '.join(tokenized_text)
        return combined_text

    def clean_data(self, train, test):
        # applying the cleaning function to both test and training datasets
        train['text_clean'] = train['text'].apply(str).apply(lambda x: self.text_preprocessing(x))
        test['text_clean'] = test['text'].apply(str).apply(lambda x: self.text_preprocessing(x))

        train['text_len'] = train['text_clean'].astype(str).apply(len)
        train['text_word_count'] = train['text_clean'].apply(lambda x: len(str(x).split()))
        return train, test

