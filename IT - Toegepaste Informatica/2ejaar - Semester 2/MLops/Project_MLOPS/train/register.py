import os
import pickle
import mlflow

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


from sklearn.metrics import mean_squared_error

from prefect import task, flow
import shutil

@task
def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@task
def train_and_log_model(X_train, y_train, X_val, y_val, X_test, y_test, params, model):

    # Make copies of y_val and y_test before passing them to mean_squared_error
    X_train_copy = X_train.copy()
    y_train_copy = y_train.copy()

    y_val_copy = y_val.copy()
    y_test_copy = y_test.copy()

    X_val_copy = X_val.copy()
    X_test_copy = X_test.copy()

    if isinstance(model, KNeighborsClassifier):
        PARAMS = ['n_neighbors' ,'leaf_size', 'p']
        with mlflow.start_run():
            for param in PARAMS:
                params[param] = int(params[param]) # Convert to float

            knn = KNeighborsClassifier(**params)
            knn.fit(X_train_copy, y_train_copy)

            val_rmse = mean_squared_error(y_val_copy, knn.predict(X_val_copy), squared=False)
            mlflow.log_metric("val_rmse", val_rmse)
            test_rmse = mean_squared_error(y_test_copy, knn.predict(X_test_copy), squared=False)
            mlflow.log_metric("test_rmse", test_rmse)          

    elif isinstance(model, RandomForestClassifier):
        PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state', 'n_jobs']
        with mlflow.start_run():
            for param in PARAMS:
                params[param] = int(params[param])

            rf = RandomForestClassifier(**params)
            rf.fit(X_train_copy, y_train_copy)

            val_rmse = mean_squared_error(y_val_copy, rf.predict(X_val_copy), squared=False)
            mlflow.log_metric("val_rmse", val_rmse)
            test_rmse = mean_squared_error(y_test_copy, rf.predict(X_test_copy), squared=False)
            mlflow.log_metric("test_rmse", test_rmse)

    elif isinstance(model, SVC):
        PARAMS = ['C', 'degree', 'gamma', 'coef0', 'tol', 'cache_size', 'max_iter']
        with mlflow.start_run():
            for param in PARAMS:
                params[param] = int(params[param])

            sc = SVC(**params)
            sc.fit(X_train_copy, y_train_copy)

            val_rmse = mean_squared_error(y_val_copy, sc.predict(X_val_copy), squared=False)
            mlflow.log_metric("val_rmse", val_rmse)
            test_rmse = mean_squared_error(y_test_copy, sc.predict(X_test_copy), squared=False)
            mlflow.log_metric("test_rmse", test_rmse)

    elif isinstance(model, LogisticRegression):
        PARAMS = ['tol', 'C', 'max_iter', 'n_jobs']
        with mlflow.start_run():
            for param in PARAMS:
                params[param] = int(params[param])

            lr = LogisticRegression(**params)
            lr.fit(X_train_copy, y_train_copy)

            val_rmse = mean_squared_error(y_val_copy, lr.predict(X_val_copy), squared=False)
            mlflow.log_metric("val_rmse", val_rmse)
            test_rmse = mean_squared_error(y_test_copy, lr.predict(X_test_copy), squared=False)
            mlflow.log_metric("test_rmse", test_rmse)

    elif isinstance(model, DecisionTreeClassifier):
        PARAMS = ['max_depth', 'min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf', 'max_features', 'max_leaf_nodes', 'min_impurity_decrease', 'min_impurity_split', 'ccp_alpha']

        with mlflow.start_run():
            for param in PARAMS:
                params[param] = int(params[param])

            dc = DecisionTreeClassifier(**params)
            dc.fit(X_train_copy, y_train_copy)

            val_rmse = mean_squared_error(y_val_copy, dc.predict(X_val_copy), squared=False)
            mlflow.log_metric("val_rmse", val_rmse)
            test_rmse = mean_squared_error(y_test_copy, dc.predict(X_test_copy), squared=False)
            mlflow.log_metric("test_rmse", test_rmse)

    else:
        print("Ongeldige model")

@task
def get_experiment_runs(top_n, hpo_experiment_name):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(hpo_experiment_name)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )
    return runs

@task
def select_best_model(top_n, experiment_name):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.test_rmse ASC"]
    )[0]
    
    return best_run

@flow
def register_flow(model_path: str, top_n: int, experiment_name: str, hpo_experiment_name: str, best_model):
    mlflow.set_experiment(experiment_name)
    mlflow.sklearn.autolog()
    
    X_train, y_train = load_pickle(os.path.join(model_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(model_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(model_path, "test.pkl"))

    # Retrieve the top_n model runs and log the models
    runs = get_experiment_runs(top_n, hpo_experiment_name)
    for run in runs:
        # Assuming you have 'model' defined somewhere before this loop
        train_and_log_model(X_train, y_train, X_val, y_val, X_test, y_test, params=run.data.params, model=best_model)

    # Select the model with the lowest test RMSE
    best_run = select_best_model(top_n, experiment_name)

    # Register the best model
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, name=f"{best_model}")

    # Laad het model vanuit MLflow
    model = mlflow.sklearn.load_model(model_uri)
    # Sla het model op in de map ./models/KNeighborsClassifier
    if os.path.exists("./models/best_model") and os.listdir("./models/best_model"):
        shutil.rmtree("./models/best_model")
        mlflow.sklearn.save_model(model, path="./models/best_model")
