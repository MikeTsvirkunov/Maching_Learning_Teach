import numpy as np


class ExtendAbleBagOfWords:
    def __init__(self, bag, vector_size, uncknow_word_indicator):
        self.bag = bagiu
        self.vector_size = vector_size
        self.uncknow_word_indicator = uncknow_word_indicator
    
    def vectorize(self, words: iter):
        vector = np.array([])
        for i in range(len(words)//self.vector_size + (len(words)%self.vector_size != 0)):
            for word in words[i * self.vector_size : i * self.vector_size + min(self.vector_size, len(words) - i * self.vector_size)]:
                if word in self.bag:
                    element = np.where(self.bag == word)[0]
                else:
                    element = self.uncknow_word_indicator
                vector = np.append(vector, element)
        return vector

    def extend(self, words):
        self.bag = np.unique(np.append(self.bag, words))


class BagOfWords:
    def __init__(self, bag):
        self.bag = np.array(bag, copy=True).reshape(1, max(np.array(bag).shape))

    def vectorize(self, words: iter):
        vector = np.zeros([1, self.bag.shape[1]])
        for word in words:
            res = np.where(self.bag==word)[0]
            # print(res, word, vector[0])
            if len(res):
                # print(vector[0])
                vector[0][res[0]] += 1
                print(vector[0])
            else:
                # print('else')
                self.bag = np.append(self.bag, word)
                vector = np.append(vector[0], 1).reshape(1, vector.shape[1]+1)
        return vector



