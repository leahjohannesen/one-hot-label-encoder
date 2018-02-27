import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler

class DataFixerator(object):
    '''
    An updated version of the combo one hot encoder and label encoder.
    Currently it dummifies all columns, but this will eventually be expanded to include picking which
    columns are numeric etc.
    '''
    def __init__(self):
        self.les = []
        self.ohe = OneHotEncoder()
        self.cols = None
        self.categorical_names = {}
        self.categorical_features = []

    def fit(self, df):
        #gonna rework this to make sure the data is stored nicely
        dfc = df.copy()
        dfc = dfc.fillna('NA')
        self.cols = dfc.columns.values
        self.categorical_features = range(len(self.cols))
        self.col_map = []
       
        for i, col in enumerate(self.cols): 
            le = LabelEncoder()
            dfc.ix[:,i] = le.fit_transform(dfc.ix[:,i])
            self.les.append(le)
            self.categorical_names[i] = le.classes_
            fullmap = [(col, x) for x in le.classes_]
            self.col_map += fullmap
            
        self.ohe.fit(dfc)

    def transform(self, df):
        dfc = df.copy()
        dfc = dfc.fillna('NA')
        for i, col in enumerate(self.cols):
            dfc.ix[:,i] = self.les[i].transform(dfc.ix[:,i])

        arr = self.ohe.transform(dfc).toarray()
        res_df = pd.DataFrame(arr, index=df.index)
        return res_df 

    def get_cols(self, features):
        #given a feature set, return the corresponding columns of the full array
        idxs = sorted([i for i,c in enumerate(self.col_map) if c[0] in features])
        return idxs
