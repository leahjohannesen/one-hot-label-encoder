from src.one_hot_label_encoder import OneHotLabelEncoder
import numpy as np

if __name__ == '__main__':
    test = np.array([['a', 'man', 'fl', 'brown'], 
                     ['b', 'woman', 'fl', 'blonde'],
                     ['a', 'woman', 'tx', 'black'],
                     ['c', 'woman', 'ak', 'red']])
    
    ohle = OneHotLabelEncoder()
    ohle.fit(test)
    result = ohle.transform(test)
    print result
