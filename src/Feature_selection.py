import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fnmatch
#Sklearn libraries
from sklearn.linear_model import LassoCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold

def unique_count(target_df):
    for cols in target_df:
        counter = target_df[cols].value_counts()
        print("\n" + cols + ":")
        print(counter)

def split_column(full_df, split_col, new_cols = ["col1", "col2"], split_by = ", "):
    cols = full_df.columns.tolist()
    pos = full_df.columns.get_loc(split_col)
    df1 = full_df[split_col].str.split(split_by, expand=True)
    df1.columns = new_cols
    cols[pos:pos] = df1.columns.tolist()
    cols.remove(split_col)
    full_df = full_df.join(df1).reindex(cols, axis=1)
    return full_df

def rfecv_features(model, X, y , n , m = 5, scoring_metrics='accuracy'):
    if model == 'dt': model_input = DecisionTreeClassifier()
    elif model =='rf': model_input = RandomForestClassifier()
    else:
        print('invalid model input, Random Forest Model use as default')
        model_input = RandomForestClassifier()
    rfecv_model = RFECV(estimator=model_input, cv=StratifiedKFold(10), scoring=scoring_metrics, min_features_to_select=n, step=m,verbose=1, n_jobs=-1)
    rfecv_model.fit(X,y)
    feat_ls = [feat for feat, result in list(zip(X, rfecv_model.support_)) if result == True ]
    feat_coeff = rfecv_model.estimator_.feature_importances_
    print('no.of features = ',len(feat_ls))
    df_feat_importance = pd.DataFrame()
    df_feat_importance['features'] = feat_ls
    df_feat_importance['importance'] = feat_coeff
    df_feat_importance = df_feat_importance.sort_values(by=f'importance', ascending=True)
    plt.figure(figsize=(10, 10))
    plt.barh(y=df_feat_importance['features'], width=df_feat_importance[f'importance'])
    plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)
    plt.xlabel(f'importance', fontsize=14, labelpad=20)
    plt.show()
    df_feat_importance.columns = ['features',f'{model}-importance']
    return df_feat_importance, feat_ls, feat_coeff

def selectkbest_features(X,y,classifier=f_classif,k=30):
    select_k = SelectKBest(classifier, k = 30)
    fit = select_k.fit(X, y)
    df_selectkbest = pd.DataFrame()
    df_selectkbest['features'] = X.columns.tolist()
    df_selectkbest['k_best'] = fit.scores_
    df_selectkbest.sort_values('k_best', ascending=False, inplace = True)
    df_selectkbest.plot(kind ='barh')
    return df_selectkbest

def lasso_features(X,y):
    lasso_model = LassoCV()
    print(lasso_model.get_params)
    lasso_model.fit(X, y)
    print('lasso_cv score = ',lasso_model.score(X,y))
    df = pd.DataFrame()
    df['features'] = X.columns
    df['lasso'] = abs(lasso_model.coef_)
    df.sort_values('lasso', ascending=False, inplace = True)
    if df[df['lasso']!= 0].shape[0] >0:
        df_display = df.copy()
        df_display.set_index('features', inplace =True)
        display(df_display[df_display['lasso']!= 0].head())
        df_display[df_display['lasso']!= 0].plot.barh()
        plt.show()
    return df

def merge_info(df_feature, df_input):
    df_feature = pd.merge(df_feature, df_input, on='features', how='left')
    return df_feature

def corr(X,y):
    target = y.columns[0]
    X_corr = pd.concat([X, y], axis=1)
    df_corr = X_corr.corr()
    df_corr_target = df_corr[target].abs().sort_values(ascending = False).to_frame()
    df_corr_target.drop(index=target,inplace=True)
    df_corr_target.reset_index(inplace=True)
    df_corr_target.columns = ['features', 'corr']
    display(df_corr_target.head())
    return df_corr_target

def run_fs(X,y, n = 30):
    indicators = X.columns.to_list()
    df_features = pd.DataFrame(indicators, columns=['features'])  

    df_corr=corr(X,y)
    df_features = merge_info(df_features, df_corr)
    df_selectkbest = selectkbest_features(X, y)
    df_features = merge_info(df_features, df_selectkbest)
    df_lasso = lasso_features(X, y)
    df_features = merge_info(df_features, df_lasso)
    df_rf_rfecv, _ , _ = rfecv_features('rf', X, y, n=n, m = 2)
    df_features = merge_info(df_features, df_rf_rfecv)
    df_dt_rfecv, _ , _ = rfecv_features('dt', X, y, n=n, m = 2)
    df_features = merge_info(df_features, df_dt_rfecv)

    for i in df_features.columns[1:]:
        df_features[f'{i}_rank'] = df_features[i].rank(ascending = False)
        df_features[f'{i}_rank'] = df_features[f'{i}_rank'].apply(lambda x : 1 if x <= n else 0)
    rank_ls = fnmatch.filter(df_features.columns,'*_rank')
    df_features['voting'] = df_features[rank_ls].sum(axis=1)
    df_features.sort_values('voting', ascending=False, inplace = True)
    df_features.reset_index(drop=True, inplace=True)
    
    return df_features
