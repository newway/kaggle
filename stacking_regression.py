import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso,  BayesianRidge, LassoCV,RidgeCV, Ridge, PassiveAggressiveRegressor, SGDRegressor
from sklearn import svm, tree, neighbors, gaussian_process
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor,BaggingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
meta_lasso = Lasso(alpha =0.05, random_state=1)
meta_ridge = Ridge(alpha =0.05, random_state=1)
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        if models is not None:
            self.models = models
        else:
            self.models = (ENet, GBoost, KRR, lasso, model_xgb)
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        #start = time.time()
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)
        #self.__fittime__ = time.time() - start
        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, cv=5):
        if base_models is None:
            self.base_models = (ENet, GBoost, KRR, model_xgb)
        else:
            self.base_models = base_models
        if meta_model is None:
            self.meta_model = meta_lasso 
        else:
            self.meta_model = meta_model
        self.cv = cv

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.cv, shuffle=True, random_state=156)
        #kfold = self.cv
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_ = self.meta_model_.fit(out_of_fold_predictions, y)
        try:
            coef = pd.Series(self.meta_model_.coef_, index = [ m.__class__.__name__ for m in self.base_models ])
            print("stack index coef:", coef)    #, coef[coef!=0].index
        except (AttributeError):
            pass
        return self

    #Do the predictions of all base models on the test data and use the averaged predictions as
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

REG = [
    #GLM - remove linear models, since this is a regressor algorithm
    ('brr', BayesianRidge()),
    ('enet', ElasticNet()),
    ('lasso', Lasso()),
    ('lr', LinearRegression()),
    ('par', PassiveAggressiveRegressor()),
    ('ridge', Ridge()),
    ('sgd', SGDRegressor()),

    #Gaussian Processes
    ('gpr', gaussian_process.GaussianProcessRegressor()),

    #Kernel Ridge Regressor
    ('krr', KernelRidge()),

    #Ensemble Methods
    ('ada', AdaBoostRegressor(tree.DecisionTreeRegressor())),
    ('bag', BaggingRegressor()),
    ('etr',ExtraTreesRegressor()),
    ('gbr', GradientBoostingRegressor()),
    ('xgbr',xgb.XGBRegressor(max_depth=3)),  # xgb.XGBRegressor()),    #
    ('rfr', RandomForestRegressor(n_estimators = 50)),

    #Nearest Neighbor
    ('knr', neighbors.KNeighborsRegressor(n_neighbors = 3)),

    #SVM
    ('svr', svm.SVR(kernel='rbf', gamma=0.1)),
    ('lsvr', svm.LinearSVR()),

    #Trees
    ('dtr', tree.DecisionTreeRegressor()),
    ('etr2', tree.ExtraTreeRegressor()),
]
ESTS_PARAM_GRID = {
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
    'ridge':    [{'alpha':[10],'tol':[0.0001],'solver':['auto'], 'random_state':[1]}]
}

MLA_columns = ['Name', 'Test Score Mean', 'Train Score Mean', 'Test Train Diff', 'Test Score +-3Std', 'Parameters', 'Selected Columns', 'Fit Time', 'TrainData','TestData','Mode']
def ensemble_regression(data, target, test, all_columns, featureFromModel=False, cv_split=5, score_f='neg_mean_squared_error', EST_PARAM_GRID=ESTS_PARAM_GRID, **kw):
    #cv_split = model_selection.ShuffleSplit(n_splits = n_splits, test_size = testpart, train_size = trainpart, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10
    #create table to compare MLA
    MLA_compare = pd.DataFrame(columns = MLA_columns)
    #index through REG and save performance to table
    row_index = 0
    base_ests = []
    for est in REG:
        name, alg = est
        if name not in EST_PARAM_GRID:
            continue
        print(name)
        if featureFromModel:
            sfMode, selected_columns = get_feature_by_mode(alg, data, target, all_columns)
            MLA_compare.loc[row_index, 'TrainData'] = sfMode.transform(data)
            MLA_compare.loc[row_index, 'TestData'] = sfMode.transform(test)
            MLA_compare.loc[row_index, 'Selected Columns'] = [ all_columns[i] for i in selected_columns]
        else:
            MLA_compare.loc[row_index, 'TrainData'] = data
            MLA_compare.loc[row_index, 'TestData'] = test
            MLA_compare.loc[row_index, 'Selected Columns'] = all_columns.tolist()
        param_grid = EST_PARAM_GRID[name]
        n_features = len(MLA_compare.loc[row_index]['Selected Columns'])
        for pg in param_grid:
            if 'max_features' in pg:
                max_feature_l = pg['max_features']
                pg['max_features'] = [x for x in max_feature_l if not isinstance(x, int) or x<=n_features ]       #filter(lambda x: x<=len(n_features), max_feature_l)

        tune_model = GridSearchCV(alg, param_grid=param_grid, scoring = score_f, cv=cv_split, return_train_score=True, n_jobs= 4)
        tune_model = tune_model.fit(MLA_compare.loc[row_index, 'TrainData'], target)
        cv_results = tune_model.cv_results_
        best_index = tune_model.best_index_     # index of best param
        MLA_compare.loc[row_index, 'Parameters'] = str(tune_model.best_params_).replace(" ","")
        MLA_compare.loc[row_index, 'Name'] = alg.__class__.__name__
        MLA_compare.loc[row_index, 'Mode'] = tune_model.best_estimator_
        try :
            MLA_compare.loc[row_index, 'Mode'].coef_
            coef = pd.Series(tune_model.best_estimator_.coef_, index = all_columns)
            MLA_compare.loc[row_index, 'Selected Columns'] = coef[coef!=0].index.tolist()
            print(name," coef >0 :", len(MLA_compare.loc[row_index, 'Selected Columns']))
        except (AttributeError):
            pass
        MLA_compare.loc[row_index, 'Fit Time'] = cv_results.get('mean_fit_time')[best_index]
        MLA_compare.loc[row_index, 'Train Score Mean'] = cv_results.get('mean_train_score')[best_index]
        MLA_compare.loc[row_index, 'Test Score Mean'] = cv_results.get('mean_test_score')[best_index]   # == tune_model.best_score_
        MLA_compare.loc[row_index, 'Test Train Diff'] = rmsle(target, MLA_compare.loc[row_index, 'Mode'].predict(data)) #MLA_compare.loc[row_index, 'Test Score Mean'] - MLA_compare.loc[row_index, 'Train Score Mean']
        MLA_compare.loc[row_index, 'Test Score +-3Std'] = (-MLA_compare.loc[row_index, 'Test Score Mean']-3*cv_results.get('std_test_score')[best_index], -MLA_compare.loc[row_index, 'Test Score Mean']+3*cv_results.get('std_test_score')[best_index])   #let's make +-3Std minus for ascending order

        base_ests.append((name, tune_model.best_estimator_))
        row_index+=1

    print('average ests number:', len(base_ests))
    if len(base_ests) >= 2:
        avg_base = [y for x,y in base_ests]
        alg = AveragingModels(models=avg_base)
        alg.fit(data, target)
        MLA_compare.loc[row_index, 'Parameters'] = 'avg{}'.format(len(base_ests))
        MLA_compare.loc[row_index, 'Name'] = 'AverageModel'
        MLA_compare.loc[row_index, 'Mode'] = alg
        MLA_compare.loc[row_index, 'Selected Columns'] = all_columns.tolist()
        #MLA_compare.loc[row_index, 'Fit Time'] = alg.__fittime__
        cv_results = cross_validate(alg, data, target, scoring = score_f, cv=cv_split, return_train_score=True, n_jobs= 4)
        MLA_compare.loc[row_index, 'Fit Time'] = cv_results['fit_time'].mean()
        MLA_compare.loc[row_index, 'Train Score Mean'] = cv_results['train_score'].mean()
        MLA_compare.loc[row_index, 'Test Score Mean'] = cv_results['test_score'].mean()
        test_score_std = cv_results['test_score'].std()
        MLA_compare.loc[row_index, 'Test Train Diff'] = rmsle(target, MLA_compare.loc[row_index, 'Mode'].predict(data)) # MLA_compare.loc[row_index, 'Test Score Mean'] - MLA_compare.loc[row_index, 'Train Score Mean']
        MLA_compare.loc[row_index, 'Test Score +-3Std'] = (-MLA_compare.loc[row_index, 'Test Score Mean']-3*test_score_std, -MLA_compare.loc[row_index, 'Test Score Mean']+3*test_score_std)
        MLA_compare.loc[row_index, 'TestData'] = test
        row_index+=1
        stack_models = []
        stack_base = {'enet':{}, 'gbr':{}, 'krr':{}, 'rfr':{}}
        meta_model = RandomForestRegressor(n_estimators=100, random_state=1) #  #RandomForestRegressor(n_estimators=100, random_state=1, max_depth=3) # LassoCV(alphas=[0, 0.1, 0.01, 0.0005, 1, 5]) #
        for x,y in base_ests:
            if x in stack_base:
                stack_base[x] = y
                stack_models.append(y)
        #stack_models = [y for x,y in base_ests[1:]]
        #meta_model = base_ests[0][1]
        print("StackingAveragedModels base model:", len(stack_models), "\nmeta model:", meta_model)
        alg = StackingAveragedModels(base_models=stack_models, meta_model=meta_model)
        alg.fit(data, target)
        MLA_compare.loc[row_index, 'Parameters'] = str(alg.meta_model_.get_params())
        MLA_compare.loc[row_index, 'Name'] = 'StackingAveragedModels'
        MLA_compare.loc[row_index, 'Mode'] = alg
        MLA_compare.loc[row_index, 'Selected Columns'] = all_columns.tolist()
        #MLA_compare.loc[row_index, 'Fit Time'] = alg.__fittime__
        cv_results = cross_validate(alg, data, target, scoring = score_f, cv=cv_split, return_train_score=True, n_jobs= 4)
        MLA_compare.loc[row_index, 'Fit Time'] = cv_results['fit_time'].mean()
        MLA_compare.loc[row_index, 'Train Score Mean'] = cv_results['train_score'].mean()
        MLA_compare.loc[row_index, 'Test Score Mean'] = cv_results['test_score'].mean()
        test_score_std = cv_results['test_score'].std()
        MLA_compare.loc[row_index, 'Test Train Diff'] = rmsle(target, MLA_compare.loc[row_index, 'Mode'].predict(data)) # MLA_compare.loc[row_index, 'Test Score Mean'] - MLA_compare.loc[row_index, 'Train Score Mean']
        MLA_compare.loc[row_index, 'Test Score +-3Std'] = (-MLA_compare.loc[row_index, 'Test Score Mean']-3*test_score_std, -MLA_compare.loc[row_index, 'Test Score Mean']+3*test_score_std)
        MLA_compare.loc[row_index, 'TestData'] = test
        row_index+=1

    # get the best mode
    MLA_compare[['Train Score Mean','Test Score Mean']] = -1*MLA_compare[['Train Score Mean','Test Score Mean']]
    MLA_compare.sort_values(by = ['Test Score Mean', 'Test Train Diff'], ascending = True, inplace = True)
    #MLA_compare.reset_index(drop=True, inplace = True)
    print_columns = MLA_columns[:-4]
    print(MLA_compare[print_columns], '\n')
    for i in range(row_index):
        print(MLA_compare.iloc[i]["Parameters"])
    feature_name = "_FM{}" if featureFromModel else "_fk{}"
    ensemble_name = MLA_compare.iloc[0]['Name'][:5] + MLA_compare.iloc[0]['Parameters'].replace(':','_') + feature_name.format(len(MLA_compare.iloc[0]['Selected Columns']))
    model = MLA_compare.iloc[0]["Mode"]
    test_data = MLA_compare.iloc[0]['TestData']
    #prediction = np.asarray(MLA_compare.iloc[row_index-2]['Mode'].predict(test_data))*0.5 + np.asarray(MLA_compare.iloc[row_index-1]['Mode'].predict(test_data))*0.5 
    prediction = MLA_compare.iloc[0]['Mode'].predict(test_data)
    #avg_weights = [0.6,0.1,0.1,0.1,0.1]
    #prediction = np.sum(np.column_stack([avg_weights[i]*np.asarray(MLA_compare.loc[i, 'Mode'].predict(test_data)) for i in range(row_index)]),axis=1)
    return (model, prediction, ensemble_name)

