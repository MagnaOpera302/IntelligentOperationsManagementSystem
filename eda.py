import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

sale_df = pd.read_csv("./ML Datasets/SuperstoreSalesData/train.csv")

print(sale_df.shape)
print(sale_df.head())
print(sale_df.sample(10))