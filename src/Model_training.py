import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from xgboost.sklearn import XGBClassifier

def pipeline_grid( X_train, X_test, y_train, y_test, model , scaler,):
    
    full_name_dict = {
                      'lr'       : 'Logistic Regression',
                      'rf'       : 'Random Forest Classifier', 
                      'xgb'      : 'XG Boost Classifier',
                      }

    scaler_dict ={'mm' : MinMaxScaler(),
                  'ss' : StandardScaler(),
                  }
    
    param_dict = {'lr' : {'lr__C':[0.1,1,5,10,25,50],
                           'lr__max_iter':[1000,2000, 3000]},
          
                  'rf' : {'rf__n_estimators': [100],
                          'rf__max_depth': [None, 1, 2],
                          'rf__min_samples_split': [5,10],
                          'rf__min_samples_leaf': [2,3]},

                  'xgb':{'xgb__max_depth':[2,4],
                         'xgb__n_estimators':[100, 500],
                         'xgb__min_child_weight': [1, 10],
                         'xgb__subsample': [0.6, 0.8],
                         'xgb__colsample_bytree': [0.6,1.0],
                         'xgb__learning_rate' : [0.1, 0.1]}
                  }

    model_dict = {
                  'lr' : LogisticRegression(solver='lbfgs'), 
                  'rf' : RandomForestClassifier(),
                   'xgb' : XGBClassifier(objective = 'binary:logistic',
                                        #early_stopping_rounds=200,
                                        #metric ='aucpr', 
                                        #silent=False,
                                        )
                  }

    #choosing model for modelling
    model_dict_1 ={
                  'lr' : LogisticRegression,
                  'rf' : RandomForestClassifier,
                   'xgb': XGBClassifier,
                  }
    
    pipe = Pipeline([(scaler, scaler_dict[scaler]), (model, model_dict[model])])
    
    
    pipe_params = param_dict[model]
    grid = GridSearchCV(pipe,
           param_grid=pipe_params,
           cv=StratifiedKFold(n_splits=10),n_jobs=-1,scoring = "roc_auc", verbose = 1)
        
    grid.fit(X_train, y_train.values.ravel())
    # scores = pd.DataFrame(grid.cv_results_['params'])
    test_score_grid = grid.score(X_test,y_test)
    print('Test score from grid',test_score_grid)

    print('Best score on training data',round(grid.best_score_,4))    
    print('Best Params',grid.best_params_)   
    string_remove = model + '__'
    parameter = dict((k.replace(string_remove,''),v) for k,v in grid.best_params_.items())

    choosen_model = model_dict_1[model](**parameter).fit(X_train,y_train)   
    print('*******Test Data*******')
    pred = choosen_model.predict(X_test)
    roc_auc_score(y_test,pred)

    return pred

def assess_model_simple(y_axis, pred):
    f1 = f1_score(y_true = y_axis, y_pred = pred, average = "binary")

    tn, fp, fn, tp = confusion_matrix(y_axis, pred).ravel()
    accuracy          = round((tp+tn)/(tp+fp+tn+fn),4)*100
    recall            = round(tp/(tp+fn),4)

    return f1, recall, accuracy

def assess_model(y_axis, pred):
    
    f1 = f1_score(y_true = y_axis, y_pred = pred, average = "binary")
    c_matrix = confusion_matrix(y_pred = pred, y_true = y_axis)

    tn, fp, fn, tp = confusion_matrix(y_axis, pred).ravel()
    
    accuracy          = round((tp+tn)/(tp+fp+tn+fn),4)*100
    misclassification = round((fp+fn)/(tp+fp+tn+fn),4)*100
    precision         = round(tp/(tp+fp),4) * 100
    recall            = round(tp/(tp+fn),4)
    specificity       = round(tn/(tn+fp),4)
    
    print('** Accuracy %: {}% **'     .format(accuracy))
    print(f"f1 score: {f1}\n")
    print('Misclassification %: {}&'.format(misclassification))
    print('Precision %: {}'           .format(round(tp/(tp+fp),4)))
    print('Recall: {}'                .format(round(tp/(tp+fn),4)))
    print('Specificity: {}'           .format(round(tn/(tn+fp),4)))


    print("####### END OF REPORT #######")

    return f1, accuracy, misclassification , precision, recall, specificity


def grid_searcher( X_train, X_test, y_train, y_test, model, scaler,):
    
    full_name_dict = {
                      'lr'       : 'Logistic Regression',
                      'rf'       : 'Random Forest Classifier', 
                      'xgb'      : 'XG Boost Classifier',
                      }

    scaler_dict ={'mm' : MinMaxScaler(),
                  'ss' : StandardScaler(),
                  }
    
    param_dict = {'lr' : {'lr__C':[0.1,1,5,10,25,50],
                           'lr__max_iter':[250,500,1000]},
          
                  'rf' : {'rf__n_estimators': [100],
                          'rf__max_depth': [None, 1, 2],
                          'rf__min_samples_split': [5,10],
                          'rf__min_samples_leaf': [2,3]},

                  'xgb':{'xgb__max_depth':[2,4],
                         'xgb__n_estimators':[100, 500],
                         'xgb__min_child_weight': [1, 10],
                         'xgb__subsample': [0.6, 0.8],
                         'xgb__colsample_bytree': [0.6,1.0],
                         'xgb__learning_rate' : [0.1, 0.1]}
                  }

    model_dict = {
                  'lr' : LogisticRegression(solver='lbfgs'), 
                  'rf' : RandomForestClassifier(),
                   'xgb' : XGBClassifier(objective = 'binary:logistic',
                                        #early_stopping_rounds=200,
                                        #metric ='aucpr', 
                                        #silent=False,
                                        )
                  }

    #choosing model for modelling
    model_dict_1 ={
                  'lr' : LogisticRegression,
                  'rf' : RandomForestClassifier,
                   'xgb': XGBClassifier,
                  }
    
    pipe = Pipeline([(scaler, scaler_dict[scaler]), (model, model_dict[model])])
    
    
    pipe_params = param_dict[model]
    grid = GridSearchCV(pipe,
           param_grid=pipe_params,
           cv=StratifiedKFold(n_splits=10),n_jobs=-1,scoring ='roc_auc', verbose = 1)
        
    grid.fit(X_train, y_train.values.ravel())
    # scores = pd.DataFrame(grid.cv_results_['params'])
    test_score_grid = grid.score(X_test,y_test)
    print('Test score from grid',test_score_grid)

    print('Best score on training data',round(grid.best_score_,4))    
    print('Best Params',grid.best_params_)   
    string_remove = model + '__'
    parameter = dict((k.replace(string_remove,''),v) for k,v in grid.best_params_.items())
    
    choosen_model = model_dict_1[model](**parameter).fit(X_train,y_train)   
    print('*******Test Data*******')
    pred = choosen_model.predict(X_test)
    roc_auc_score(y_test,pred)
    print_out(full_name_dict[model],X_test,y_test,choosen_model)

    return choosen_model

#Functions to Plot ROC curve with AUC scoring
# Define function to calculate sensitivity. (True positive rate.)
def TPR(df, true_col, pred_prob_col, threshold):
    true_positive = df[(df[true_col] == 1) & (df[pred_prob_col] >= threshold)].shape[0]
    false_negative = df[(df[true_col] == 1) & (df[pred_prob_col] < threshold)].shape[0]
    return true_positive / (true_positive + false_negative)

# Define function to calculate 1 - specificity. (False positive rate.)
def FPR(df, true_col, pred_prob_col, threshold):
    true_negative = df[(df[true_col] == 0) & (df[pred_prob_col] <= threshold)].shape[0]
    false_positive = df[(df[true_col] == 0) & (df[pred_prob_col] > threshold)].shape[0]
    return 1 - (true_negative / (true_negative + false_positive))

def Calculate(pred_df,thresholds):
    # Calculate sensitivity & 1-specificity for each threshold between 0 and 1.
    tpr_values = [TPR(pred_df, 'true_values', 'pred_probs', prob) for prob in thresholds]
    fpr_values = [FPR(pred_df, 'true_values', 'pred_probs', prob) for prob in thresholds]
    return tpr_values,fpr_values

def ROC_curve(y_test, X_test,model):
    
    pred_proba = [i[1] for i in model.predict_proba(X_test)]
    print(pred_proba)
    pred_df = pd.DataFrame({'true_values': y_test,
                           'pred_probs': pred_proba})

    roc_auc = roc_auc_score(pred_df['true_values'], pred_df['pred_probs'])
    print('ROC-AUC SCORE:',roc_auc)
    GINI = (2 * roc_auc) - 1
    print('Gini score:',GINI)
   
    # Create figure.
    plt.figure(figsize = (10,7))

    # Create threshold values. (Dashed red line in image.)
    thresholds = np.linspace(0, 1, 200)
    
    tpr_values,fpr_values = Calculate(pred_df,thresholds)
    
    # Plot ROC curve.
    plt.plot(fpr_values, # False Positive Rate on X-axis
             tpr_values, # True Positive Rate on Y-axis
             label='ROC Curve')

    # Plot baseline. (Perfect overlap between the two populations.)
    plt.plot(np.linspace(0, 1, 200),
             np.linspace(0, 1, 200),
             label='baseline',
             linestyle='--')

    # Label axes.
    plt.title(f'ROC Curve with AUC = {round(roc_auc_score(pred_df["true_values"], pred_df["pred_probs"]),3)}', fontsize=22)
    plt.ylabel('Sensitivity', fontsize=18)
    plt.xlabel('1 - Specificity', fontsize=18)

    # Create legend.
    plt.legend(fontsize=16);
    plt.show()  

def confusion(X_test,y_test,model):
    display(pd.concat([y_test.value_counts(normalize=True),y_test.value_counts()],axis=1 ))
    pred = model.predict(X_test)

    ## Edit here Female = 0 , Male = 1
    df_confusion = pd.DataFrame(confusion_matrix(y_test,pred), 
             index = ['actual {}'.format('Female'), 'actual {}'.format('Male')],
             columns = ['Predict  {}'.format('Female'), 'Predict  {}'.format('Male')]
            )
    display(df_confusion)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    
    accuracy          = round((tp+tn)/(tp+fp+tn+fn),4)*100
    misclassification = round((fp+fn)/(tp+fp+tn+fn),4)*100
    precision         = round(tp/(tp+fp),4) * 100
    recall            = round(tp/(tp+fn),4)
    specificity       = round(tn/(tn+fp),4)
    
    print('** Accuracy %: {}% **'     .format(accuracy))
    print('Misclassification %: {}&'.format(misclassification))
    print('Precision %: {}'           .format(round(tp/(tp+fp),4)))
    print('Recall: {}'                .format(round(tp/(tp+fn),4)))
    print('Specificity: {}'           .format(round(tn/(tn+fp),4)))
    
    return accuracy, misclassification , precision, recall, specificity
              
def print_out(gs,X_test,y_test,model):
    print('''
    ###########################
    {} model
    ###########################    
    '''.format(gs))
    accuracy, misclassification , precision, recall, specificity = confusion(X_test,y_test,model)
    
    ROC_curve(y_test,X_test,model)
    