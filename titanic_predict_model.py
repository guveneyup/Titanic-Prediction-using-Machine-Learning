#################################
# VALUES
#################################
# PassengerId: Id
# Survived: hayatta kalma(0 = Hayır, 1 =Evet)
# Pclass: bilet sınıfı(1 = 1., 2 = 2., 3 = 3.)
# Sex: cinsiyet
# Sibsp: Titanik’teki kardeş/eş sayısı
# Parch:Titanik’teki ebeveynlerin/çocukların sayısı
# Ticket: bilet numarası
# Fare: ücret
# Cabin: kabin numarası
# Embarked: biniş limanı
# Name: İsim

#################################
# LIBRARIES
#################################
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.exceptions import ConvergenceWarning
import warnings
#################################
# Data set settings
#################################
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

#################################
# READ DATA SET
#################################
train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")
combine = [train_df, test_df]


#################################
# FUNCTIONS
#################################


def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)

    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    print(missing_df, end="\n")

    if na_name:
        return na_columns


def outlier_thresholds(dataframe, col_name):
    quartile1 = dataframe[col_name].quantile(0.05)
    quartile3 = dataframe[col_name].quantile(0.95)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if low_limit > 0:
        dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit
    else:
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit


def label_encoder(dataframe, binary_col):
    labelencoder = preprocessing.LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


#################################
# EDA
#################################


for dataset in combine:
    print(check_df(dataset))
    print("-" * 30)

for dataset in combine:
    print(dataset.quantile([0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1]).T)
    print("-" * 30)

columns = [col for col in train_df.columns if train_df[col].dtypes != "O" and train_df[col].nunique() > 10]
for col in columns:
    sns.histplot(x=dataset[col])
    plt.title("DENSITY")
    plt.xlabel(col)
    plt.show()

#################################
# MISSING VALUES
#################################


for i in combine:
    print(i.isnull().values.any())
    print("-" * 20)

for i in combine:
    print(i.isnull().sum())
    print("-" * 20)

# How many deficiencies are observed in the variables with missing observation values and the ratio within itself

for dataset in combine:
    print(dataset.shape)
    missing_values_table(dataset, na_name=True)
    print("-" * 30)


# Let me investigate whether the missing values of the variables containing missing observation
# values are important according to the dependent variable.
# Since there is no TARGET variable in the test_df data set, it will be taken from the train_df data set.


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


na_cols = missing_values_table(train_df, na_name=True)
missing_vs_target(train_df, "Survived", na_cols)

# Missing values in the CABIN variable have less chance of being salvaged on average.
#################################
# FILL IN MISSING OBSERVATIONS
#################################

for dataset in combine:
    dataset["Age"].fillna(dataset.groupby("Sex")["Age"].transform("median"), inplace=True)

train_df["Embarked"].fillna(train_df["Embarked"].dropna().mode()[0], inplace=True)

test_df["Fare"].fillna(test_df["Fare"].dropna().median(), inplace=True)

for dataset in combine:
    dataset["NEW_CABIN_BOOL"] = dataset["Cabin"].isnull().astype('int')

for dataset in combine:
    dataset.drop("Cabin", axis=1, inplace=True)

for dataset in combine:
    print(dataset.shape)
    print(dataset.isnull().values.any())

######################
# Outlier Values
######################

train_df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T

num_cols = [i for i in train_df.columns if train_df[i].dtypes != "O" and train_df[i].nunique() > 10]

for i in num_cols:
    print(i, check_outlier(train_df, i))

replace_with_thresholds(train_df, "Fare")


for i in num_cols:
    print(i, check_outlier(train_df, i))
###########################
# Local Outlier Factor
###########################
dff = train_df.select_dtypes(include=['float64', 'int64'])
clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(dff)
df_scores = clf.negative_outlier_factor_

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 20], style='.-')
np.sort(df_scores)[0:10]
plt.show()
esik_deger = np.sort(df_scores)[3]
train_df.drop(labels=train_df[df_scores < esik_deger].index, axis=0, inplace=True)

#############################################
# CATEGORICAL VS TARGET ANALYSIS
#############################################

train_cat_cols = [i for i in train_df.columns if train_df[i].nunique() < 5 and i not in "Survived"]

for i in train_cat_cols:
    print(train_df.groupby(i).agg({"Survived": ["mean", "count"]}))
    print("-" * 20)

#############################################
# NUMERICAL VS TARGET ANALYSIS
#############################################

for i in num_cols:
    print(train_df.groupby("Survived").agg({i: ["mean"]}))

#############################################
# FEATURE EXTRACTION
#############################################

for dataset in combine:
    dataset.loc[((dataset['SibSp'] + dataset['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
    dataset.loc[((dataset['SibSp'] + dataset['Parch']) == 0), "NEW_IS_ALONE"] = "YES"

for dataset in combine:
    dataset["NEW_NAME_COUNT"] = dataset["Name"].str.len()
    dataset["NEW_NAME_WORD_COUNT"] = dataset["Name"].apply(lambda x: len(str(x).split(" ")))
    dataset["NEW_FAMILY_SIZE"] = dataset["SibSp"] + dataset["Parch"] + 1

for dataset in combine:
    dataset["NEW_AGExPCLASS"] = dataset["Age"] * dataset["Pclass"]

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# COLUMNS UPPER
for dataset in combine:
    dataset.columns = [col.upper() for col in dataset.columns]

########################
# Label Encoding
########################

for dataset in combine:
    binary_cols = [col for col in dataset.columns if len(dataset[col].unique()) == 2 and dataset[col].dtypes == 'O']
    for i in binary_cols:
        label_encoder(dataset, i)

print(f"Train: {train_df.shape} \nTest: {test_df.shape}")

##########################
# One-Hot Encoding
##########################

ohe_cols = [col for col in train_df.columns if 5 >= len(train_df[col].unique()) > 2]
train_df = one_hot_encoder(train_df, ohe_cols)

ohe_cols = [col for col in test_df.columns if 5 >= len(test_df[col].unique()) > 2]
test_df = one_hot_encoder(test_df, ohe_cols)

print(f"Train: {train_df.shape} \nTest: {test_df.shape}")

######################
# SET VARIABLE
######################
train_df.drop(['TICKET', 'NAME', "PASSENGERID"], axis=1, inplace=True)
X = train_df.drop("SURVIVED", axis=1)
y = train_df["SURVIVED"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)


X_test = test_df.drop(['PASSENGERID', 'TICKET', 'NAME'], axis=1).copy()
y_test = pd.read_csv("dataset/gender_submission.csv")
y_test.drop("PassengerId", axis=1, inplace=True)

######################
# MODELS
######################
models = {"RANDOM FORREST CLASS": RandomForestClassifier(),
          "XGBOOST CLASS": XGBClassifier(),
          "LGBM CLASS": LGBMClassifier(),
          "GRADIENT BOOSTING CLASS": GradientBoostingClassifier()}

for model in models.keys():
    fit = models[model].fit(X_train, y_train)
    print(f"{model} : {accuracy_score(y_val, fit.predict(X_val))}")

######################
# MODEL TUNING
######################

# RANDOM FORREST TUNING
rf_params = {"max_depth": [3, 5, 8, None],
             "max_features": [3, 5, 15, 20],
             "n_estimators": [1000, 2000],
             "min_samples_split": [2, 5, 8]}

rfm_model = RandomForestClassifier(random_state=123)
rfm_cv_model = GridSearchCV(rfm_model, rf_params, cv=5, n_jobs=-1, verbose=1).fit(X_train, y_train)

rf_tuned = RandomForestClassifier(**rfm_cv_model.best_params_).fit(X_train, y_train)
y_pred = rf_tuned.predict(X_val)
accuracy_score(y_val, y_pred)


# XGBOOST CLASS TUNING
xgb = XGBClassifier().fit(X_train, y_train)
xgb_params = {"learning_rate": [0.1, 0.01],
              "max_depth": [5, 8],
              "n_estimators": [100, 1000],
              "colsample_bytree": [0.7, 1]}

xgb_cv_model = GridSearchCV(xgb, xgb_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)


xgb_tuned = XGBClassifier(**xgb_cv_model.best_params_).fit(X_train, y_train)
y_pred = xgb_tuned.predict(X_val)
np.sqrt(accuracy_score(y_val, y_pred))


# LGBM CLASS TUNING


lgb_model = LGBMClassifier()

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1000],
               "max_depth": [3, 5, 8],
               "colsample_bytree": [1, 0.8, 0.5]}

lgbm_cv_model = GridSearchCV(lgb_model,
                             lgbm_params,
                             cv=10,
                             n_jobs=-1,
                             verbose=2).fit(X_train, y_train)


lgbm_tuned = LGBMClassifier(**lgbm_cv_model.best_params_).fit(X_train, y_train)
y_pred = lgbm_tuned.predict(X_val)
np.sqrt(accuracy_score(y_val, y_pred))


# GRADIENT BOOSTING CLASS TUNING

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8],
              "n_estimators": [500, 1000],
              "subsample": [1, 0.5, 0.7]}

gbm_model = GradientBoostingClassifier(random_state=17)
gbm_cv_model = GridSearchCV(gbm_model, gbm_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)

gbm_tuned = GradientBoostingClassifier(**gbm_cv_model.best_params_).fit(X_train, y_train)
y_pred = gbm_tuned.predict(X_val)
np.sqrt(accuracy_score(y_val, y_pred))


# CAT BOOST CLASS TUNING

catb_params = {"iterations": [200, 500],
               "learning_rate": [0.01, 0.1],
               "depth": [3, 6]}

catb_model = CatBoostClassifier()
catb_cv_model = GridSearchCV(catb_model,
                             catb_params,
                             cv=10,
                             n_jobs=-1,
                             verbose=2).fit(X_train, y_train)


catb_tuned = CatBoostClassifier(**catb_cv_model.best_params_).fit(X_train, y_train)
y_pred = catb_tuned.predict(X_val)
np.sqrt(accuracy_score(y_val, y_pred))
