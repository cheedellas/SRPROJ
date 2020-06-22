# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the KNN CLassifier model
filename = 'knnmodel.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        gen = int(request.form['gender'])
        sec8 = int(request.form['issec8'])
        mgi = int(request.form['monthlygrossincome'])
        fic = int(request.form['fico'])
        
        data = np.array([[gen, sec8, mgi, fic]])
        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)