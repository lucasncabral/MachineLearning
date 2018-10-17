import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

train = pd.read_csv("eleicoes_2006_a_2010.csv")
test = pd.read_csv("eleicoes_2014.csv")

train.head()

all_data = pd.concat((train.loc[:,'ano':'descricao_ocupacao'],
                      test.loc[:,'ano':'descricao_ocupacao']))

matplotlib.rcParams['figure.figsize'] = (17.0, 5.0)
prices = pd.DataFrame({"total_votos":train["votos"], "log(votos + 1)":np.log1p(train["votos"])})
prices.hist(log=True)


#log transform the target:
train["votos"] = np.log1p(train["votos"])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])