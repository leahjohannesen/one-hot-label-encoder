import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class OneHotLabelEncoder(object):
    '''
    This class combines the functionality of the label encoder and the one hot encoder allowing
    you to transform several categorical variables into dummy variables at once, while allowing
    you to save that transformation and apply it to later data.
    Currently it coerces all arrays passed into categorical dummy variables
    Class Parameters
        missing_dummy:
            Should a "missing value" dummy be made for each category" to allow for transformation
            of data so far unseen
        sparse:
            Should any output arrays be sparse
    Methods
        fit(x, categorical_indices='auto', col_names=None):
            Arguments
                x:
                    A numpy array of shape (n_samples x n_features)
                categorical_indices(default 'auto'):
                    If a list of indices is provide, attemps to convert those indices otherwise
                    attempts to automatically recognize if a column is categorical
                col_names(default None):
                    A list of all column names. If none, automatically labels columns with their
                    column number
                
        transform(x):
            Arguments
                x:
                    A numpy array of shape (n_samples x n_features)
       
         fit_transform(x):
            Arguments: Takes the same arguments as fit()
    Attributes
        categorical_indices_:
            List of indices that were transformed or 'auto'
        col_names_:
            List of column names of initially fitted array
        classes_:
            List of column names after Label/OneHot encoding
        le_list_:
            List of individual column LabelEncoder()
        
    '''
    def __init__(self, missing_dummy=False, sparse=False):
        self.missing_dummy_ = missing_dummy
        self.sparse_ = sparse
        self.categorical_indices_ = None
        self.col_names_ = None
        self.classes_ = []
        self.le_list_ = []
        
        self._new_col_names = []
        self._fit_shape = None
        self._ohe_ = None


    def _make_le(self, row, idx):
        row_le = LabelEncoder()        

        row_xformed = row_le.fit_transform(row)
        le_cats = row_le.classes_
        return row_xformed, row_le, le_cats

    def _trans_le(self, row, idx):
        row_le = self.le_list_[idx]
        row_xform = row_le.transform(row)
        return row_xform

    def _convert_col_names(self):
        new_classes = []
        for idx in xrange(len(self.col_names_)):
            new_classes.extend(map(lambda x: str(idx) + '_' + x, self._new_col_names[idx]))
        self.classes_ = new_classes

    def fit(self, x, categorical_indices='auto', col_names=None):
        self.categorical_indices_ = categorical_indices
        self.col_names_ = col_names

        try:
            type(x) == type(np.array([]))
        except:
            print "Type of x needs to be numpy array"
        
        self._fit_shape = x.shape
        
        if not self.col_names_:
            self.col_names_ = range(self._fit_shape[1])

        x_trans = x.T
        array_le = np.zeros(x_trans.shape)
        for i in range(self._fit_shape[1]):
            row_enc, le, new_col_classes = self._make_le(x_trans[i], i)
            self.le_list_.append(le)
            array_le[i] = row_enc
            self._new_col_names.append(new_col_classes)     
        self._ohe_ = OneHotEncoder(sparse=self.sparse_)
        self._ohe_.fit(array_le.T)
        
        self._convert_col_names()
        
        return

    def transform(self, x):
        if not x.shape[1] == self._fit_shape[1]:
            print "Number of columns fit != number of columns to transform"
            return
        x_trans = x.T
        array_le = np.zeros(x_trans.shape)
        for i in range(x.shape[1]):
            row_enc = self._trans_le(x_trans[i], i)
            array_le[i] = row_enc
        
        x_enc = self._ohe_.transform(array_le.T)
        return x_enc

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
