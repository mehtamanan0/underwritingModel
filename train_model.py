import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib

# load the final training data
train_df = pd.read_csv("data/train_data_preprocessed.csv")

# load the training values in a numpy format
X = train_df.values[:, 0:21].astype(float)

# load the labels
Y = train_df.values[:, 21].astype(int)

model = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0,
                                   criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1,
                                   min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0,
                                   min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=0,
                                   max_leaf_nodes=None, warm_start=False, presort='auto')

model.fit(X, Y)

joblib.dump(model, 'models/model')
