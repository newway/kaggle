import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
from subprocess import check_output
from scipy.stats import skew
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.special import boxcox1p
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
import lightgbm as lgb
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, LabelEncoder, normalize, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, Lasso, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, ShuffleSplit
import xgboost as xgb
import sys
sys.path.append('..')
from stacking_regression import AveragingModels, StackingAveragedModels, model_xgb, model_lgb, meta_lasso, meta_ridge, ENet, GBoost, KRR, ensemble_regression

def DataFrameLoad():
    print(check_output(["ls", "input/"]).decode("utf8"))
    # Any results you write to the current directory are saved as output.
    train = pd.read_csv("input/train.csv", header=0)
    test = pd.read_csv("input/test.csv", header=0)
    # drop outliers
    #print(train[['OverallQual', 'SalePrice']].groupby(['OverallQual']).describe())
    #print(train[['GrLivArea', 'SalePrice']].groupby(['GrLivArea']).describe())
    train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index, inplace=True)
    train.drop(train[(train['OverallQual']<5) & (train['SalePrice']>200000)].index, inplace=True)
    train.drop(train[(train['OverallQual']<=2) & (train['SalePrice']>=60000)].index, inplace=True)
    train.drop(train[(train['OverallQual']==10) & (train['SalePrice']<170000)].index, inplace=True)
    #print(train[['OverallQual', 'SalePrice']].groupby(['OverallQual']).describe())
    #exit()
    #print(train.info())     #1460 r
    #print(test.info())      #1459 r
    #train.drop(miss_delete_columns, axis=1, inplace=True)
    feature_to_use = [ col for col in train.columns if col in test.columns]
    combined = train[feature_to_use].append(test[feature_to_use])
    get_missing_rate(combined)
    #print(combined.info())
    # nearly the same value in all samples, no use
    combined.drop(['Utilities'], axis=1, inplace=True)
    return (combined, train, test)

def process_MS(X):
    #maybe replace missing value with most relative feature's corresponding value
    #print(X[['MSZoning', 'MSSubClass']])
    group_MSSubClass = X[['MSSubClass','MSZoning']].groupby(['MSSubClass'])
    group_MSSubClass_mode = group_MSSubClass.apply(lambda x: x.mode())
    X['MSZoning'] = X.apply(lambda r: group_MSSubClass_mode.loc[r['MSSubClass']]['MSZoning'][0] if pd.isnull(r['MSZoning']) else r['MSZoning'], axis=1)
    #print(X[['MSZoning', 'MSSubClass']])
    return X

def process_garage(X):
    X.GarageCond.fillna('None', inplace=True)
    X.GarageQual.fillna('None', inplace=True)
    X.GarageFinish.fillna('None', inplace=True)
    X.GarageType.fillna('None', inplace=True)
    X.GarageYrBlt.fillna(0, inplace=True)
    X.GarageArea.fillna(0, inplace=True)
    X.GarageCars.fillna(0, inplace=True)
#X.garage was comprehensived of all garage factors 

def process_Bsmt(X):
    X.BsmtCond.fillna('None', inplace=True)
    X.BsmtQual.fillna('None', inplace=True)
    X.BsmtExposure.fillna('None', inplace=True)
    X.BsmtFinType1.fillna('None', inplace=True)
    X.BsmtFinType2.fillna('None', inplace=True)
    X.BsmtHalfBath.fillna(0, inplace=True)
    X.BsmtFullBath.fillna(0, inplace=True)
    X.BsmtFinSF2.fillna(0, inplace=True)
    X.BsmtFinSF1.fillna(0, inplace=True)
    X.BsmtUnfSF.fillna(0, inplace=True)
    X.TotalBsmtSF.fillna(0, inplace=True)

def process_MasVnr(X):
    X.MasVnrType.fillna('None', inplace=True)
    X.MasVnrArea.fillna(0, inplace=True)

def process_Exterior(X):
    X.Exterior2nd.fillna(X.Exterior2nd.mode()[0], inplace=True)
    X.Exterior1st.fillna(X.Exterior1st.mode()[0], inplace=True)


def DataImpute(X):
    #X.PoolQC.fillna("None", inplace=True)
    #X.MiscFeature.fillna("None", inplace=True)
    #X.Alley.fillna("None", inplace=True)
    #X.Fence.fillna("None", inplace=True)
    X.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1, inplace=True)
    X = process_MS(X)
    X.FireplaceQu.fillna("None", inplace=True)
    #X.LotFrontage = X.groupby('Neighborhood')["LotFrontage"].transform(lambda x:x.fillna(x.median()))
    X.LotFrontage.fillna(X.LotFrontage.mean(), inplace=True)
    process_garage(X)
    process_Bsmt(X)
    process_MasVnr(X)
    process_Exterior(X)
    X.Electrical.fillna(X.Electrical.mode()[0], inplace=True)
    X.Functional.fillna('Typ', inplace=True)
    X.SaleType.fillna(X.SaleType.mode()[0], inplace=True)
    X.KitchenQual.fillna(X.KitchenQual.mode()[0], inplace=True)
    if X.isnull().any().any():
        print('Nan After Imputer, exit!', X.isnull().any())
        exit()
    return X

def get_missing_rate(X):
    #missing data
    total = X.isnull().sum().sort_values(ascending=False)
    percent = (X.isnull().sum()/X.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    #print(missing_data.head(35))
    miss_delete_columns = missing_data[missing_data['Percent']>0.8].axes[0].tolist()
    #print(miss_delete_columns)
    return miss_delete_columns

def DataMangle(X):
    ##convert digital column which is really category
    X['MSSubClass'] = X['MSSubClass'].astype(str)
    X['OverallCond'] = X['OverallCond'].astype(str)
    #X['YrSold'] = X['YrSold'].astype(str)
    #X['MoSold'] = X['MoSold'].astype(str)
    X['YrSold'] = X['YrSold'] + X['MoSold']/12.0
    # add one column total square feet
    X.drop(["Id",'MoSold'], axis = 1, inplace = True)
    #normalizing data
    numeric_feat = X.dtypes[X.dtypes != "object"].index
    skewed_feats = X[numeric_feat].apply(lambda x: skew(x.dropna()))    #return a series
    skewed_feats = skewed_feats[(skewed_feats) > 0.75].index
    # log transform make skewed feature more normal
    #X[skewed_feats] = np.log1p(X[skewed_feats])
    lam = 0.15
    X[skewed_feats] = boxcox1p(X[skewed_feats], lam)
    # Adding total sqfootage feature
    X[numeric_feat] = RobustScaler().fit_transform(X[numeric_feat]) #
    return X

def map_category_by_target(train, test, tg_name, target, c):
    train[tg_name] = target
    #print(train.head(2))
    uniq = train[c].unique().tolist()
    print(uniq)
    def fit_cagegory_by_target(train, c, tg_name):
        #print(train.groupby([c]).head(2))
        grp = train.groupby([c])[tg_name].mean()
        #print(grp)
        code_map = {}   
        for v in uniq:
            #print(v, grp.loc[v][tg_name])
            code_map[v] = int(grp[v]*10)
        return code_map
    code_map = fit_cagegory_by_target(train, c, tg_name)
    for x in test[c].unique():
        if x not in code_map:
            # maybe need manually filled
            code_map[x] = code_map[train[c].mode()[0]]
    return code_map

def DataEncoder(X):
    # ordered category
    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'OverallCond')   #,'OverallQual', 'YrSold', 'MoSold','PoolQC', 'MiscFeature', 'Alley', 'Fence'
    #cols = X.select_dtypes(include = ["object"]).columns
    # process columns, apply LabelEncoder to categorical features
    for c in cols:
        #cate_map = map_category_by_target(X[:train.shape[0]], X[train.shape[0]:], 'SalePrice', train['SalePrice'], c)
        #X[c] = X[c].map(cate_map)
        lbl = LabelEncoder()
        X[c] = lbl.fit_transform(X[c])
    X = pd.get_dummies(X)
    print(X.shape)
    return X

def rmse_cv(model, data, target, cv):
    kf = KFold(cv, shuffle=True, random_state=42)   #.get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, data, target, scoring="neg_mean_squared_error", cv = kf, n_jobs=-1))
    return (rmse)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def get_K_corr(X, y, k=20):
    train_corr = X.copy() 
    train_corr['target-'] = y
    corrmat = train_corr.corr()     #train_corr.corrwith(y)
    cor_feat = corrmat['target-'].drop(labels='target-').sort_values(axis=0, ascending=False)
    #X.drop(cor_feat[abs(cor_feat) < 0.15].index, axis=1, inplace=True)
    cor_feat = cor_feat[abs(cor_feat) > 0.15]
    print(cor_feat, len(cor_feat))
    #return cor_feat[abs(cor_feat) < 0.15].index
    return cor_feat.index.tolist()[:k]

combined, train, test = DataFrameLoad()
train['SalePrice'] = np.log1p(train['SalePrice'])
target = train['SalePrice'].as_matrix()
test_Id = test.Id
DataImpute(combined)
# why log of target:price, because price is a multiple mode of all factors, log transform make it linear
combined = DataMangle(combined)
train_df = combined[:train.shape[0]]
test_df = combined[train.shape[0]:]
#key_fe = get_K_corr(train_df, train['SalePrice'], k=25)
#combined.drop(key_fe, axis=1, inplace=True)
combined = DataEncoder(combined)
all_columns = combined.columns
train_M = combined[:train.shape[0]].as_matrix()
test_M = combined[train.shape[0]:].as_matrix()
#print(np.any(np.isinf(train_M)), np.any(np.isinf(target)))

score_f = make_scorer(rmsle, greater_is_better=False)
param_house = {
    'lasso':    [{'alpha': [0.0005], 'random_state':[1]}],     # first model is used for meta model in StackingAveragedModels
    'xgbr':     [{'colsample_bytree':[0.4603], 'gamma':[0.0468],
                    'learning_rate':[0.05], 'max_depth':[3],
                    'min_child_weight':[1.7817], 'n_estimators':[2200],
                    'reg_alpha':[0.4640], 'reg_lambda':[0.8571],
                    'subsample':[0.5213], 'silent':[1],
                    'random_state':[7], 'nthread':[-1]}],
    'enet':     [{'alpha':[0.0005], 'l1_ratio':[.9], 'random_state':[3]}],
    'krr':      [{'alpha':[0.6], 'kernel':['polynomial'], 'degree':[2], 'coef0':[2.5]}],
    'gbr':      [{'n_estimators':[3000], 'learning_rate':[0.05],
                                   'max_depth':[4], 'max_features':['sqrt'],
                                   'min_samples_leaf':[15], 'min_samples_split':[10],
                                   'loss':['huber'], 'random_state':[5]}],
    #'rfr': [{'n_estimators':[3000], 'max_features':[0.46],'min_samples_leaf':[3],'min_samples_split':[6]}],
    #'ridge':    [{'alpha':[10],'tol':[0.0001],'solver':['auto'], 'random_state':[1]}]
}
cv_split = ShuffleSplit(n_splits = 10, test_size = 0.3, train_size = 0.6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10
#cv_split = KFold(5, shuffle=True, random_state=42)
model,prediction,m_name = ensemble_regression(train_M, target, test_M, all_columns, cv_split=cv_split, score_f=score_f, EST_PARAM_GRID=param_house)    #, EST_PARAM_GRID=param_grid
submission = pd.DataFrame({"Id": test_Id, "SalePrice": np.expm1(prediction)})
subname = m_name + '.csv'
submission.to_csv(subname, index=False)
