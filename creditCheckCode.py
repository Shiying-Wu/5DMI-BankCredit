# Data handling
import numpy as np
import pandas as pd

# visualization
import matplotlib.pyplot as plt #Basic plotting

# warning control
import warnings
warnings.filterwarnings("ignore")

#decision tree 
from sklearn.tree import DecisionTreeClassifier,plot_tree

#Neural network
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense


df = pd.read_csv("./credit copy 1.csv")


df.info()

