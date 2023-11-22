from flask import Flask, render_template, request
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

car_evaluation = fetch_openml(name='car', version=3)

X = car_evaluation.data
y = car_evaluation.target

categorical_cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                le = LabelEncoder()
                output[col] = le.fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                le = LabelEncoder()
                output[colname] = le.fit_transform(col)
        return output

def train_and_evaluate_model(classifier, X_train, y_train, X_test, y_test):
    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer([('label_encoder', MultiColumnLabelEncoder(columns=categorical_cols), categorical_cols)], remainder='passthrough')),
        ('classifier', classifier)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)

    plot_confusion_matrix(cm, pipeline.classes_)

    return accuracy, f1, cm

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 8))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('static/confusion_matrix.png')
    plt.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    classifier_name = request.form.get('classifier')
    param1 = request.form.get('param1')
    param2 = request.form.get('param2')
    param3 = request.form.get('param3')
    param4 = request.form.get('param4')
    param5 = request.form.get('param5')
    param6 = request.form.get('param6')

    if classifier_name == 'knn':
        classifier = KNeighborsClassifier(n_neighbors=5)
    elif classifier_name == 'svm':
        classifier = SVC(C=1.0, kernel='linear')
    elif classifier_name == 'mlp':
        classifier = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)
    elif classifier_name == 'dt':
        classifier = DecisionTreeClassifier(max_depth=None)
    elif classifier_name == 'rf':
        classifier = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)

    accuracy, f1, cm = train_and_evaluate_model(classifier, X_train, y_train, X_test, y_test)

    return render_template('results.html', accuracy=accuracy, f1=f1, confusion_matrix=cm)

if __name__ == '__main__':
    app.run(debug=True)
