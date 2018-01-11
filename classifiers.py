
import pandas as pd
import numpy as np
#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
import xgboost as xgb
#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn.feature_selection import SelectFromModel
from sklearn import model_selection
from sklearn import metrics


#Machine Learning Algorithm (MLA) Selection and initialization
CLF = [
    #Ensemble Methods
    ('ada', ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier())),
    ('bc', ensemble.BaggingClassifier()),
    ('etc',ensemble.ExtraTreesClassifier()),
    ('gbc', ensemble.GradientBoostingClassifier()),
    ('xgbc',xgb.XGBClassifier(max_depth=3)),  # xgb.XGBClassifier()),    #
    ('rfc', ensemble.RandomForestClassifier(n_estimators = 50)),

    #Gaussian Processes
    ('gpc', gaussian_process.GaussianProcessClassifier()),
    
    #GLM - remove linear models, since this is a classifier algorithm
    ('lr', linear_model.LogisticRegressionCV()),
    ('pac', linear_model.PassiveAggressiveClassifier()),
    ('rc', linear_model. RidgeClassifierCV()),
    ('sgd', linear_model.SGDClassifier()),                                                                                      
    ('pct', linear_model.Perceptron()),                                                                                          
    
    #Navies Bayes
    ('gnb', naive_bayes.GaussianNB()),
    
    #Nearest Neighbor
    ('knn', neighbors.KNeighborsClassifier(n_neighbors = 3)),
    
    #SVM
    ('svc', svm.SVC(probability=True)),
    ('lsvc', svm.LinearSVC()),
    
    #Trees    
    ('dtc', tree.DecisionTreeClassifier()),
    ('etc2', tree.ExtraTreeClassifier()),
]

EST_PARAM_GRID = {  #'ada':  [{"base_estimator__criterion": ["gini", "entropy"],         #0
                    #                        "base_estimator__splitter":   ["best", "random"],
                    #                        #'base_estimator__max_depth': [1, 2],
                    #                        #'base_estimator__min_samples_split': [5,10,20,50,70],
                    #                        #'base_estimator__max_features': [None, 'auto', 'sqrt', 'log2'],
                    #                        "learning_rate":[0.01,0.05,0.1,0.5,1],
                    #                        "n_estimators": [5,20,30,50,70],
                    #                        "algorithm": ['SAMME', 'SAMME.R'],
                    #                        "random_state":[None, 0]}],
                    #'etc':  [{  "max_depth": [None],
                    #            "max_features": [3,4,5,8,9],
                    #            "min_samples_split": [2, 3, 5, 10, 20,50],
                    #            "min_samples_leaf": [1, 3, 10,20],
                    #            "bootstrap": [False],
                    #            "n_estimators" :[30,50,100,300],
                    #            "criterion": ["gini"]}],     
                    #'gbc':  [{  'loss' : ["deviance"],
                    #            'n_estimators' : [50,100,200,300],
                    #            'learning_rate': [0.5, 0.1, 0.05, 0.01],
                    #            'max_depth': [3,4,5, 8],
                    #            'min_samples_leaf': [30,50,70,100,150],
                    #            'max_features': [0.3, 0.1,0.5]}],
                    'xgbc': [{  "max_depth":[3,4],
                                "n_estimators":[10,20,30,40,50,60,70,80,100],
                                "learning_rate":[0.01,0.05,0.1,0.5,1]}],
                    #'rfc':  [{  "max_depth": [3,4],
                    #            "max_features": [0.2,0.3,0.4,0.5,0.6,0.8],
                    #            "min_samples_split": [5, 10, 20, 50, 100],
                    #            "min_samples_leaf": [3, 5, 10, 20, 50, 100],
                    #            "bootstrap": [False],
                    #            "n_estimators" :[30,50,80,100],
                    #            "criterion": ["gini"]}],
                    #'gpc':  [{"kernel":[None], "random_state":[None, 0]}],          #no SelectFromModel http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html#sklearn.gaussian_process.GaussianProcessClassifier
                    #'lr':   [{'solver':['lbfgs', 'liblinear']}],  #http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV
                    #'gnb':  [{'priors':[None]}],                #no SelectFromModel because has no `coef_` or `feature_importances_` attribute
                    #'knn':  [{'n_neighbors':[3,5,7,11]}],          #no SelectFromModel
                    #'svc':  [{  'kernel': ['rbf'],                 #no SelectFromModel 
                    #            'gamma': [ 0.001, 0.01, 0.1, 1],   
                    #            'C': [1, 10, 50, 100,200,300, 1000]}],
                    #'dtc':  [{  'criterion': ['gini', 'entropy'],
                    #            'splitter': ['best', 'random'],
                    #            'max_depth': [None, 2,4,6,8],
                    #            'min_samples_split': [2,5,10,20,50,70],
                    #            'max_features': [None, 0.3, 'auto', 'sqrt', 'log2']}],   #auto means max_features=sqrt(n_features); None: then max_features=n_features
                    #'etc2': [{  "max_depth": [None],
                    #            "max_features": [3, 4,5,6],
                    #            "min_samples_split": [2, 3, 10, 20, 50],
                    #            "min_samples_leaf": [1, 3, 10],
                    #            "criterion": ["gini"]}],

}

def get_feature_by_mode(alg, data, target, all_columns, prefit=False, threshold=None):
    sfMode = SelectFromModel(alg, prefit=False, threshold=None)       #The estimator must have either a feature_importances_ or coef_ attribute after fitting, threshold:"0.1*mean" or 0.25 
    sfMode = sfMode.fit(data, target)
    selected_columns = sfMode.get_support(indices=True).tolist()     #list of index
    n_features = len(selected_columns)
    print(selected_columns)
    #while n_features <= 2:
    #    sfMode.threshold -= 0.1
    #    data_S = sfMode.transform(data)
    #    n_features = data_S.shape[1]
    #print('\n', name, "------SelectFromModel:", n_features, all_columns, selected_columns)
    print(alg.__class__.__name__, "------SelectFromModel:", n_features, [all_columns[x] for x in selected_columns])
    return (sfMode, selected_columns)

MLA_columns = ['MLA Name', 'MLA Test Accuracy Mean', 'MLA Train Accuracy Mean', 'Test Train Diff', 'MLA Test Accuracy -3Std', 'MLA Parameters', 'Selected Columns', 'MLA Fit Time', 'TrainData','TestData','MLA Mode']

def ensemble_model(data, target, test, all_columns, featureFromModel=False, n_splits=10, testpart=0.3, trainpart=0.6, **kw):
    #note: this is an alternative to train_test_split
    #cv_split = 3  #When the cv argument is an integer, cross_val_score uses the KFold or StratifiedKFold strategies by default,
    #cv_split = model_selection.StratifiedKFold(n_splits=10)
    cv_split = model_selection.ShuffleSplit(n_splits = n_splits, test_size = testpart, train_size = trainpart, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10

    #create table to compare MLA
    MLA_compare = pd.DataFrame(columns = MLA_columns)

    #index through MLA and save performance to table
    row_index = 0
    voting_ests = []
    for est in CLF:
        name, alg = est
        if name not in EST_PARAM_GRID:
            continue 
        
        if featureFromModel:
            sfMode, selected_columns = get_feature_by_mode(alg, data, target, all_columns)
            MLA_compare.loc[row_index, 'TrainData'] = sfMode.transform(data)
            MLA_compare.loc[row_index, 'TestData'] = sfMode.transform(test)
            MLA_compare.loc[row_index, 'Selected Columns'] = [ all_columns[i] for i in selected_columns]
        else:
            MLA_compare.loc[row_index, 'TrainData'] = data
            MLA_compare.loc[row_index, 'TestData'] = test
            MLA_compare.loc[row_index, 'Selected Columns'] = all_columns
        #print(data_S[:2,:])
        param_grid = EST_PARAM_GRID[name]
        n_features = len(MLA_compare.loc[row_index]['Selected Columns'])
        for pg in param_grid:
            if 'max_features' in pg:
                max_feature_l = pg['max_features']
                pg['max_features'] = [x for x in max_feature_l if not isinstance(x, int) or x<=n_features ]       #filter(lambda x: x<=len(n_features), max_feature_l)
        #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
        #cv_results = model_selection.cross_validate(alg, data, target, cv=3, return_train_score=True)
        #print(name, param_grid, '\n')
        tune_model = model_selection.GridSearchCV(alg, param_grid=param_grid, scoring = 'accuracy', cv=cv_split, return_train_score=True, n_jobs= 4)
        tune_model = tune_model.fit(MLA_compare.loc[row_index, 'TrainData'], target)
        #print("before rfe: ", tune_model.best_params_)
        #feature selection
        #model_rfe = feature_selection.RFECV(tune_model.best_estimator_, step = 1, scoring = 'roc_auc', cv = cv_split)
        #model_rfe.fit(data, target)
        #train_rfe = model_rfe.transform(data)
        #test_rfe = model_rfe.transform(test)
        #rfe_columns = data.columns.values[model_rfe.get_support()]
        # refit model with rfe columns
        #tune_model = tune_model.fit(train_rfe, target)
        #print("after rfe: ", tune_model.best_params_)
        cv_results = tune_model.cv_results_
        best_index = tune_model.best_index_     # index of best param
        #cv_results = model_selection.cross_validate(tune_model.best_estimator_, data, target, cv=cv_split, return_train_score=True)
        #test_means = tune_model.cv_results_['mean_test_score']  # list of all combined parameters mean  test score
        #print(test_means, np.mean(test_means), max(test_means), test_means[best_index], "std for best??:", cv_results.get('std_test_score')[best_index])
        #set name and parameters
        MLA_compare.loc[row_index, 'MLA Parameters'] = str(tune_model.best_params_).replace(" ","")
        MLA_compare.loc[row_index, 'MLA Name'] = alg.__class__.__name__
        MLA_compare.loc[row_index, 'MLA Mode'] = tune_model.best_estimator_
        MLA_compare.loc[row_index, 'MLA Fit Time'] = cv_results.get('mean_fit_time')[best_index]
        MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results.get('mean_train_score')[best_index]
        MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results.get('mean_test_score')[best_index]   # == tune_model.best_score_
        MLA_compare.loc[row_index, 'Test Train Diff'] = MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] - MLA_compare.loc[row_index, 'MLA Train Accuracy Mean']
        MLA_compare.loc[row_index, 'MLA Test Accuracy -3Std'] = -3*cv_results.get('std_test_score')[best_index]   #let's make -3Std minus for ascending order
        #MLA_compare.loc[row_index, 'Selected Columns'] = all_columns
        #MLA_compare.loc[row_index, 'TestData'] = test
        
        voting_ests.append((name, tune_model.best_estimator_))
        row_index+=1
    
    if len(voting_ests) >= 3:
        test_mean = MLA_compare['MLA Test Accuracy Mean'].mean()
        voting_ests = [voting_ests[i] for i in range(MLA_compare.shape[0]) if MLA_compare.loc[i, 'MLA Test Accuracy Mean']>=test_mean]
        print('all voting ests:\n', voting_ests)
        alg = ensemble.VotingClassifier(voting_ests)
        param_grid = {'voting':['hard', 'soft']}
        tune_model = model_selection.GridSearchCV(alg, param_grid=param_grid, scoring = 'accuracy', cv=cv_split, return_train_score=True, n_jobs=4)
        tune_model = tune_model.fit(data, target)
        cv_results = tune_model.cv_results_
        best_index = tune_model.best_index_
        #set name and parameters
        MLA_compare.loc[row_index, 'MLA Parameters'] = str(tune_model.best_params_).replace(" ","")
        MLA_compare.loc[row_index, 'MLA Name'] = alg.__class__.__name__
        MLA_compare.loc[row_index, 'MLA Mode'] = tune_model.best_estimator_
        MLA_compare.loc[row_index, 'MLA Fit Time'] = cv_results.get('mean_fit_time')[best_index]
        MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results.get('mean_train_score')[best_index]
        MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results.get('mean_test_score')[best_index]
        MLA_compare.loc[row_index, 'Test Train Diff'] = MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] - MLA_compare.loc[row_index, 'MLA Train Accuracy Mean']
        MLA_compare.loc[row_index, 'MLA Test Accuracy -3Std'] = -3*cv_results.get('std_test_score')[best_index]   #let's make -3Std minus for ascending order
        MLA_compare.loc[row_index, 'TrainData'] = data
        MLA_compare.loc[row_index, 'TestData'] = test
        MLA_compare.loc[row_index, 'Selected Columns'] = all_columns
        row_index+=1

    # get the best mode
    MLA_compare.sort_values(by = ['MLA Test Accuracy Mean', 'Test Train Diff', 'MLA Test Accuracy -3Std'], ascending = False, inplace = True)
    print_columns = MLA_columns[:-4]
    #print_columns.remove("MLA Parameters")
    print(MLA_compare[print_columns], '\n', MLA_compare.index)
    print(MLA_compare.iloc[0]["MLA Parameters"])
    #loc[0]: 原始index为0，by label;    iloc[0]:当前index为0，by postion
    #MLA_compare.head(n=1) return DataFrame, MLA_compare.loc[alg_index] return Series
    feature_name = "_F{}"
    ensemble_name = MLA_compare.iloc[0]['MLA Name'][:5] + MLA_compare.iloc[0]['MLA Parameters'].replace(':','_') + feature_name.format(len(MLA_compare.iloc[0]['Selected Columns']))
    model = MLA_compare.iloc[0]["MLA Mode"]
    test_data = MLA_compare.iloc[0]['TestData']
    return (model, model.predict(test_data), ensemble_name)
