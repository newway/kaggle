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
from sklearn.preprocessing import RobustScaler, LabelEncoder, normalize
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
    train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index, inplace=True)
    #print(train.info())     #1460 r
    #print(test.info())      #1459 r
    #train.drop(miss_delete_columns, axis=1, inplace=True)
    #miss_delete_columns = get_missing_rate(test)
    #test.drop(miss_delete_columns, axis=1, inplace=True)
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
    #X['YrSold'] = X.apply(lambda r:r['YrSold'] + r['MoSold']/12.0)
    #print(X['YrSold'].head(3))
    # add one column total square feet
    #X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
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
    #print(X.columns.tolist())
    X[numeric_feat] = RobustScaler().fit_transform(X[numeric_feat])
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
        'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'OverallCond')   #, 'YrSold', 'MoSold','PoolQC', 'MiscFeature', 'Alley', 'Fence'
    #cols = X.select_dtypes(include = ["object"]).columns
    # process columns, apply LabelEncoder to categorical features
    for c in cols:
        #cate_map = map_category_by_target(X[:train.shape[0]], X[train.shape[0]:], 'SalePrice', train['SalePrice'], c)
        #X[c] = X[c].map(cate_map)
        lbl = LabelEncoder()
        X[c] = lbl.fit_transform(X[c])

    X = pd.get_dummies(X)
    print(X.shape)
    #print(X.head(2))
    return X

def rmse_cv(model, data, target, cv):
    kf = KFold(cv, shuffle=True, random_state=42)   #.get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, data, target, scoring="neg_mean_squared_error", cv = kf, n_jobs=-1))
    return (rmse)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

combined, train, test = DataFrameLoad()
train['SalePrice'] = np.log1p(train['SalePrice'])
target = train['SalePrice'].as_matrix()
test_Id = test.Id
DataImpute(combined)
# why log of target:price, because price is a multiple mode of all factors, log transform make it linear
combined = DataMangle(combined)
combined = DataEncoder(combined)
#train_corr = pd.concat([combined[:train.shape[0]], train.SalePrice], axis=1)
#corrmat = train_corr.corr()
#cor_feat = corrmat.SalePrice.sort_values(axis=0, ascending=False)
#cor_feat = cor_feat[abs(cor_feat) > 0.15].index
#print(cor_feat, len(cor_feat))
#cor_set = set(cor_feat.tolist())
train_M = combined[:train.shape[0]].as_matrix()
test_M = combined[train.shape[0]:].as_matrix()
#print(np.any(np.isinf(train_M)), np.any(np.isinf(target)))

#alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
#cv_ridge = [rmse_cv(Ridge(alpha = alpha), train_M, target, 5).mean()
#            for alpha in alphas]
#cv_ridge = pd.Series(cv_ridge, index = alphas)
#print(cv_ridge.values, cv_ridge.min())

alpha =0.0005

#model_xgb.fit(train_M, target)
#GBoost.fit(train_M, target)
#ENet.fit(train_M, target)
#KRR.fit(train_M, target)
#score =rmse_cv(model_xgb, train_M, target, 5) 
#print(score.mean(), score.std(), score.mean()+3*score.std())
#prediction = model_xgb.predict(train_M)
#print("xgb:", rmsle(target, prediction))

#model_lgb.fit(train_M, target)
#score =rmse_cv(model_lgb, train_M, target, 5) 
#print(score.mean(), score.std(), score.mean()+3*score.std())
#prediction = model_lgb.predict(train_M)
#print("lgb:", rmsle(target, prediction))

#model_lasso = LassoCV(alphas=[0.1, 0.001, 0.0005, 0.0006, 0.0008]).fit(train_M, target)
#coef = pd.Series(model_lasso.coef_, index = combined.columns)
#lasso_pick_num = sum(coef != 0)
#lasso_pick_index = coef[coef!=0].index
#score =rmse_cv(model_lasso, train_M, target, 5)
#print("Lasso picked " + str(lasso_pick_num) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
#print(score.mean(), score.std(), score.mean()+3*score.std(), model_lasso.alpha_)
#prediction = np.expm1(model_lasso.predict(test_M))

#averaged_models = AveragingModels(models=None)
#averaged_models.fit(train_M, target)
#score = rmse_cv(averaged_models, train_M, target, 5)
#print("Averaged models score: {:.6f} ({:.6f}), worst:{:.6f}, number:{}".format(score.mean(), score.std(), score.mean()+3*score.std(), len(averaged_models.models_)))
#prediction = np.expm1(averaged_models.predict(test_M))
#print("Averaged models train rmsle:", rmsle(target, averaged_models.predict(train_M)))
#prediction = np.expm1(averaged_models.predict(test_M))

#stacked_averaged_models = StackingAveragedModels(base_models=None, meta_model=None)
#stacked_averaged_models.fit(train_M, target)
#score = rmse_cv(stacked_averaged_models, train_M, target, 5)
#weights = stacked_averaged_models.meta_model_.coef_
#print("stack-avg parameter vector (w in the cost function formula):", stacked_averaged_models.meta_model_.coef_)
#print("stacked-avg models score: {:.6f} ({:.6f}), worst:{:.6f}".format(score.mean(), score.std(), score.mean()+3*score.std()))
#train_pred = stacked_averaged_models.predict(train_M)
#print("stack meta model rmsle :", rmsle(target, train_pred))
#prediction = np.expm1(stacked_averaged_models.predict(test_M))


#xgb_pred = model_xgb.predict(test_M)
#gbt_pred = GBoost.predict(test_M)
#enet_pred = ENet.predict(test_M)
#krr_pred = KRR.predict(test_M)
#xgb_pred_e = np.expm1(xgb_pred)
#gbt_pred_e = np.expm1(gbt_pred)
#enet_pred_e = np.expm1(enet_pred)
#krr_pred_e = np.expm1(krr_pred)
#

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
model,prediction,m_name = ensemble_regression(train_M, target, test_M, combined.columns, cv_split=cv_split, score_f=score_f, EST_PARAM_GRID=param_house)    #, EST_PARAM_GRID=param_grid
submission = pd.DataFrame({"Id": test_Id, "SalePrice": np.expm1(prediction)})
subname = m_name + '.csv'
submission.to_csv(subname, index=False)
