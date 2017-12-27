# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import re
from sklearn.feature_selection import SelectKBest, f_classif, chi2
import sys
sys.path.append('..')
from classifiers import ensemble_model
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from collections import Counter

def fix_outliers(df, col):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []

    # iterate over features(columns)
    # 1st quartile (25%)
    Q1 = np.percentile(df[col], 25)
    # 3rd quartile (75%)
    Q3 = np.percentile(df[col],75)
    # Interquartile range (IQR)
    IQR = Q3 - Q1

    # outlier step
    outlier_step = 1.5 * IQR

    # Determine a list of indices of outliers for feature col
    outlier_indices = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
    print(outlier_indices)
    for i in outlier_indices:
        print("original:", df.iloc[i][col])
        df[col].loc[df[col]>df[col].mean()*3] = Q1 + 0.4*IQR
        print("after fix:", df.iloc[i][col])
    print(df.describe())
    return df

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def process_title(combined):
    # 从姓名中提取称谓的函数
    def get_title(name):
        # 正则表达式检索称谓，称谓总以大写字母开头并以句点结尾
        title_search = re.search(' ([A-Za-z]+)\.', name)
        # 如果称谓存在则返回其值
        if title_search:
            return title_search.group(1)
        return ""

    # 将每个称谓映射到一个整数，有些太少见的称谓可以压缩到一个数值
    #mlle:miss, Master:院长 Rev:牧师 Major:少将 Col:上校 Mme:Mrs, Don:大学教师 Layd: 贵族女眷或夫人 
    #Countess:女伯爵 Jonkheer:最低等级的贵族（男） Sir:爵士#{"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 5, "Major": 5, "Col": 5,
    #"Mlle": 2, "Mme": 3, "Don": 6, "Lady": 6, "Countess": 6, "Jonkheer": 6, "Sir": 6, "Capt": 5, "Ms": 3, "Dona": 6}
    def map_title(title):
        title_mapping = {
                            "Capt":       "Officer",
                            "Col":        "Officer",
                            "Major":      "Officer",
                            "Jonkheer":   "Royalty",
                            "Don":        "Royalty",
                            "Sir" :       "Royalty",
                            "Dr":         "Officer",
                            "Rev":        "Officer",
                            "Countess":   "Royalty",
                            "Dona":       "Royalty",
                            "Mme":        "Mrs",
                            "Mlle":       "Miss",
                            "Ms":         "Mrs",
                            "Mr" :        "Mr",
                            "Mrs" :       "Mrs",
                            "Miss" :      "Miss",
                            "Master" :    "Master",
                            "Lady" :      "Royalty"
            }
        if title in title_mapping:
            return title_mapping[title]
        print(title, "not in title_mapping")
        return 0

    combined["Title"] = combined["Name"].apply(get_title).apply(map_title)

def process_age(combined, grouped_median_age):
    # a function that fills the missing values of the Age variable
    def fillAges(row, grouped_median):
        return grouped_median.loc[row['Sex'], row['Pclass'], row['Title']]['Age']
    combined.Age = combined.apply(lambda r : fillAges(r, grouped_median_age) if np.isnan(r['Age']) 
                                                      else r['Age'], axis=1)

def process_cabin(combined, grouped_mode_cabin):
    # a function that fills the missing values of the Age variable
    def fillCabin(row, grouped_mode):
        return grouped_mode.loc[row['Pclass']]['Cabin'][0]
    combined.Cabin = combined.apply(lambda r :  fillCabin(r, grouped_mode_cabin) if pd.isnull(r['Cabin']) else r['Cabin'], axis=1)       #1 or ‘columns’: apply function to each row


#process missing data
def TitanicImputer(X):
    print("before Imputer\n", X.describe())
    #Embarked
    X['Embarked'].fillna(X['Embarked'].mode()[0], inplace=True)
    #Fare
    X['Fare'].fillna(X['Fare'].median(), inplace=True)
    #Title
    process_title(X)
    #Age
    grouped_age = X.groupby(['Sex','Pclass','Title'])
    grouped_median_age = grouped_age.median()
    #print(grouped_median_age)
    process_age(X, grouped_median_age)
    #Cabin
    grouped_cabin = X.groupby(['Pclass'])   #returned DataFrameGroupBy
    grouped_mode_cabin = grouped_cabin.apply(lambda x:x.mode()) #returned DataFrame
    #print(grouped_mode_cabin.loc[1]['Cabin'][0])
    #print(grouped_mode_cabin.loc[2]['Cabin'][0])
    #print(grouped_mode_cabin.loc[3]['Cabin'][0])
    process_cabin(X, grouped_mode_cabin)
    #print("check NaN after Imputer:\n", X.isnull().any())
    if X.isnull().any().any():
        print('Nan After Imputer, exit!', X.isnull().any())
        exit()
    return X

#clean data, and extract features 
def DataFrameMangle(X):
    #global mangled_columns
    X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
    X['IsAlone'] = 1
    X['IsAlone'].loc[X['FamilySize']>1] = 0
    #X['Fare'] = fix_outliers(X, 'Fare')
    #qcut: 按区间内的样本数均分，cut: 按区间均分
    X["FareBin"] = pd.qcut(X["Fare"], 4, labels=False)  #X["FareBin"] = pd.qcut(X["Fare"], 4, labels=["a", "b", "c", "d"]/False)
    X["AgeBin"] = pd.cut(X["Age"].astype(int), 5, labels=False)
    X.drop('Name', axis=1, inplace=True)
    X.drop('Fare', axis=1, inplace=True)
    X.drop('Age', axis=1, inplace=True)
    return X

#encode label, adjust to estimator
def DataFrameEncode(X):
    def OneHotEncodeCategory(X):
        for col in X:
            #print(col, X[col].dtype, is_number(X[col].iloc[0]))
            if not is_number(X[col].iloc[0]):
                col_dummies = pd.get_dummies(X[col], prefix=col)
                X = pd.concat([X, col_dummies], axis=1)
                X.drop(col, axis=1, inplace=True)
        return X
    def LabelEncodeCategory(X):
        le = LabelEncoder()     # this can only be used for tree-based estimator, because of no sense of order
        for col in X:
            if not is_number(X[col].iloc[0]):
                X[col] = le.fit_transform(X[col])
        return X
    X["Sex"] = X["Sex"].map({'male':1,'female':0})
    X["Cabin"] = X["Cabin"].map(lambda c:c[0])
    #X = OneHotEncodeCategory(X)
    X = LabelEncodeCategory(X)     #only for tree estimator
    #print("after encode: head,tail 2rows\n", X.head(2), X.tail(2))
    return X

def DataFrameLoad():
    print(check_output(["ls", "input/"]).decode("utf8"))
    # Any results you write to the current directory are saved as output.
    train = pd.read_csv("input/train.csv", header=0)
    test = pd.read_csv("input/test.csv", header=0)
    #print(train.info())
    #print(test.info())
    feature_to_use = ["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"]
    combined = train[feature_to_use].append(test[feature_to_use])
    #print(train.describe())
    #print(test.describe())
    #print(combined.describe())
    return (combined, train, test)

#与模型无关的选择特征
def get_k_best_feature(train, target, K=10):
    selector = SelectKBest(f_classif, k=K)   #http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest
    selector.fit(train, target)
    # 得到每个特征列的p值，再转换为交叉验证得分
    scores = -np.log10(selector.pvalues_)
    print(sorted(scores, reverse=True), scores.mean(), '\n--------------')
    selector_feature = np.argsort(scores, axis=0)[:-(K+1):-1] 
    selected_columns = [mangled_columns[x] for x in selector_feature]
    print([(mangled_columns[col], scores.tolist()[col]) for col in selector_feature])
    print("\nselected columns:", K)
    print(selected_columns)
    return (selector, selected_columns)

combined, train, test = DataFrameLoad()
target = train['Survived']
test_pid = test['PassengerId']
combined_imputed = TitanicImputer(combined)
combined_mangled = DataFrameMangle(combined_imputed)
combined_encoded = DataFrameEncode(combined_mangled)
train = combined_encoded[:train.shape[0]]
test = combined_encoded[train.shape[0]:]
mangled_columns = combined_encoded.columns.values.tolist()
print("train/test rows: {}, {}, prepared columns: {}\n".format(train.shape[0], test.shape[0], len(mangled_columns)), mangled_columns)

train_M = train.as_matrix()
test_M = test.as_matrix()

featureFromModel = False
if not featureFromModel: 
    selected_feature_num = 10
    if selected_feature_num > len(mangled_columns):
        print("featue num exceed prepared!")
        exit()
    selector,selected_columns = get_k_best_feature(train, target, selected_feature_num)
    combined_selected = selector.transform(combined_encoded)
    train_S = combined_selected[:train.shape[0]]
    test_S = combined_selected[train.shape[0]:]

    model,prediction,m_name = ensemble_model(train_S, target, test_S, selected_columns, featureFromModel)
else:
    model,prediction,m_name = ensemble_model(train_M, target, test_M, mangled_columns, featureFromModel)
    train_S = train_M 
    test_S = test_M
    selected_feature_num = "fromMode"

#m_name = "xgbc_"
#model = xgb.XGBClassifier(max_depth=8, n_estimators=100, learning_rate=0.05)
#model.fit(train_S, train_y)
#prediction = model.predict(test_S)

#model must have fit method
scores = cross_val_score(model, train_S, target, cv=3)
print(scores)
print(scores.mean())

submission_name = m_name + '.csv'
submission = pd.DataFrame({"PassengerId": test_pid, "Survived": prediction})
submission.to_csv(submission_name, index=False)
