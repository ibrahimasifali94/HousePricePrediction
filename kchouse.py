#Importing the Libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the Dataset 
dataset = pd.read_csv('kc_house_data.csv')
hprice = dataset
y = dataset["price"]

