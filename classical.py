from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def models(seed=1234):
    return {
      'SVC':Pipeline([('scaler',StandardScaler()),('model',SVC(kernel='rbf',C=10,gamma='scale',probability=True,random_state=seed))]),
      'DTC':DecisionTreeClassifier(criterion='gini',max_depth=10,min_samples_split=5,random_state=seed),
      'LR':Pipeline([('scaler',StandardScaler()),('model',LogisticRegression(solver='liblinear',C=1,max_iter=1000,random_state=seed))]),
      'RFC':RandomForestClassifier(n_estimators=300,max_depth=15,criterion='gini',random_state=seed,n_jobs=-1),
      'K-NN':Pipeline([('scaler',StandardScaler()),('model',KNeighborsClassifier(n_neighbors=7,weights='distance',metric='minkowski',p=2))])}
