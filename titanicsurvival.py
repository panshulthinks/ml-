import pandas as pd

#lets import some algorithims also
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.plotting import scatter_matrix

# lets load our data
data = pd.read_csv(r"C:\Users\Panshul\Desktop\panshul\download\Titanic-Dataset.csv") # we used r cus there was a unicode error

# lets analyze our data
print(data.columns.values)
print(data.head())

# we can see ticket countains both numerical and categorical type of data

