import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class OneHotLabelEncoder(object):
    '''
    This class combines the functionality of the label encoder and the one hot encoder allowing
    you to transform several categorical variables into dummy variables at once, while allowing
    you to save that transformation and apply it to later data.

    Methods
        fit(x):
            Accepts a numpy array of shape (n_samples, n_categorical_features)
        transform(x):
            Acecpts a numpy array of shape (n_samples, n_categorical_features) where the
            number of categorical features matches the fitted number of categorical feat
        fit_transform(x):
            Fits and transforms an array of shape (n_samples, n_categorical_features)

    Arguments
        col_names_:
            list of column names of the whole array being passed
        missing_dummy_:
            creates a dummy for missing variables in possible transformed data
            otherwise ignores "new values" when transforming

    '''
    def __init__(self, categorical_indices='auto', col_names=None, missing_dummy=False,
                 sparse=False):
        self.categorical_indices_ = categorical_indices
        self.col_names_ = col_names
        self.classes_ = []
        self.missing_dummy_ = missing_dummy
        self.le_list_ = []
        self.sparse_ = sparse
        
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

    def fit(self, x):
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
