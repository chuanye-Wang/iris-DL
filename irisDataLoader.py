import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
import torch

class iris_data_loader(DataLoader):
    def __init__(self, path):
        self.dataset = pd.read_csv(path,names=["sepal length", "sepal width", "petal length", "petal width", "cat"])
        self.feature = self.dataset.iloc[:,:4]

        mapping = {
            "Iris-setosa":0,
            "Iris-versicolor":1,
            "Iris-virginica":2    
        }

        self.dataset["cat"] = self.dataset["cat"].map(mapping)
        self.label = self.dataset.iloc[:,4:]

        self.feature = (self.feature - self.feature.mean()) / self.feature.std()
        
        self.feature = torch.from_numpy(np.array(self.feature, dtype='float32')) 
        self.label = torch.from_numpy(np.array(self.label, dtype='int64'))

        
    def __len__(self):
        return len(self.label)


    def __getitem__(self, i):
        return self.feature[i], self.label[i]
        

if __name__ == "__main__":
    path = "iris\iris.csv"
    iris = iris_data_loader(path)
    # print(iris.__getitem__(3))
