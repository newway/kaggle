import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC, Ridge
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
meta_lasso = Lasso(alpha =0.0005, random_state=1)
meta_ridge = Ridge(alpha =0.0005, random_state=1)
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
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        if base_models is None:
            self.base_models = (ENet, GBoost, KRR, model_xgb)
        else:
            self.base_models = base_models
        if meta_model is None:
            self.meta_model = meta_lasso 
        else:
            self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

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
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    #Do the predictions of all base models on the test data and use the averaged predictions as
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)

#params = [{'max_depth':[1,2,3], 'learning_rate':[0.01,0.05,0.1,0.5,1], 'n_estimators':[100,200,300,500,1000]}]
#tune_model = GridSearchCV(xgb.XGBRegressor(), param_grid=params, scoring='neg_mean_squared_error', cv=5, return_train_score=True, n_jobs=4).fit(train_M, target)
#cv_results = tune_model.cv_results_
#best_index = tune_model.best_index_
#best_param = tune_model.best_params_
#model_xgb = tune_model.best_estimator_
#train_mean_score = -cv_results.get('mean_train_score')[best_index]
#test_mean_score = -cv_results.get('mean_test_score')[best_index]
#test_std_score = cv_results.get('std_test_score')[best_index]
#print("best model:-----------", model_xgb)
#print(best_index, best_param)
#print("\ntrain_mean_score:", train_mean_score)
#print("test_mean_score", test_mean_score, "worst +3std:", test_mean_score+3*test_std_score)
#print("test-train diff score", test_mean_score-train_mean_score, "test std:", test_std_score)
#xgb_predict = np.expm1(model_xgb.predict(test_M))
#submission = pd.DataFrame({"Id": test.Id, "SalePrice": xgb_predict})
#subname = "xgbreg_{}_{}_{}.csv".format(best_param['max_depth'], best_param['n_estimators'], best_param['learning_rate'])


