import os
import pickle
import mlflow
from prefect import task, flow


from sklearn.ensemble import RandomForestRegressor



from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

@task
def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@task
def start_ml_experiment(X_train, y_train, model):

    X_train = X_train.copy()
    y_train = y_train.copy()

    if isinstance(model, KNeighborsClassifier):
        with mlflow.start_run():
            knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
            knn.fit(X_train, y_train)
    elif isinstance(model, SVC):
        with mlflow.start_run():
            svc = SVC(kernel='linear', C=1.0)
            svc.fit(X_train, y_train)
    elif isinstance(model, RandomForestClassifier):
        with mlflow.start_run():
            rf = RandomForestClassifier(n_estimators=100, criterion='gini')
            rf.fit(X_train, y_train)
    elif isinstance(model, LogisticRegression):
        with mlflow.start_run():
            lr = LogisticRegression(solver='lbfgs', C=1.0)
            lr.fit(X_train, y_train)
    elif isinstance(model, DecisionTreeClassifier):
        with mlflow.start_run():
            dc = DecisionTreeClassifier(max_depth=5, criterion='gini')
            dc.fit(X_train, y_train)
    else:
        print("Ongeldige model")



@flow
def train_flow(model_path: str, best_model):
    mlflow.set_experiment("MLops-project-train")
    mlflow.sklearn.autolog()
    
    X_train, y_train = load_pickle(os.path.join(model_path, "train.pkl"))

    print(best_model)


    start_ml_experiment(X_train, y_train, best_model)
