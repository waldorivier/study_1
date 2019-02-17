import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.dummy import DummyRegressor
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from scipy.linalg import lstsq
from scipy import stats

from sklearn.metrics import r2_score
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

import itertools
import math
import random as r

import  PIL
from PIL import Image

working_dir = os.getcwd()
working_dir = os.path.join(working_dir,'course_projects', 'module_4')
  



