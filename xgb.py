def get_xgb(seed=1234):
    from xgboost import XGBClassifier
    return XGBClassifier(booster='gbtree',learning_rate=.1,n_estimators=200,max_depth=6,random_state=seed,eval_metric='logloss')
