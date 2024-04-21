from flask import Flask, render_template, request
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

iris_dataset = pd.read_csv('Iris_flower_dataset.csv')

X = iris_dataset.drop(['Id', 'Species'], axis=1)
y = iris_dataset['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        
        user_input = [float(request.form['sepal_length']),
                      float(request.form['sepal_width']),
                      float(request.form['petal_length']),
                      float(request.form['petal_width'])]

        
        user_predictions = knn_classifier.predict([user_input])

        return render_template('index.html', prediction=f"Predicted Species: {user_predictions[0]}")
    else:
        return render_template('index.html', prediction="Error predicting species. Please check your input.")

if __name__ == '__main__':
    app.run(debug=True)