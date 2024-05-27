import os
import pickle
import mlflow
from prefect import task, flow
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from load.prep import read_dataset_to_csv




@task
def define_x_and_y(dataset):
    X = dataset.drop(columns=['Quality'])
    y = dataset['Quality']
    return X, y


@task
def split_train_test_val(dataset):
    X = dataset.drop(columns=['Quality'])
    y = dataset['Quality']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2
    
    return X_train, X_test, X_val, y_train, y_test, y_val


@task
def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)



@task 
def training_pipeline_cross_val(X,y):

    # Define classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(),
        'KNN': KNeighborsClassifier()
    }

    # Define pipeline and perform cross-validation
    results = {}
    best_score = 0
    best_model = None


    for clf_name, clf in classifiers.items():
        pipeline = Pipeline([('clf', clf)])
        cv_scores = cross_val_score(pipeline, X, y, cv=5)
        results[clf_name] = cv_scores.mean()
            
            # Check if current classifier outperforms previous best model
        if cv_scores.mean() > best_score:
            best_score = cv_scores.mean()
            best_model = clf_name   
        
    best_model = classifiers[best_model]
    return best_model

@flow
def model_prep_flow(dest_path: str):

    print("register flow")
    #mlflow.set_experiment(experiment_name)
    #mlflow.sklearn.autolog()

    dataset = read_dataset_to_csv("processed_dataset.csv")
    x_train, X_test, X_val, y_train, y_test, y_val = split_train_test_val(dataset)
    
    dump_pickle((x_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))

    beste_model = training_pipeline_cross_val(x_train,y_train)

    return beste_model
    