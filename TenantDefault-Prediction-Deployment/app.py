# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the RF CLassifier model
filename = 'rfmodel.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        gen = int(request.form['gender'])
		mgi = int(request.form['MonthlyGrossIncome'])
		fic = int(request.form['FICO'])
		agev = int(request.form['Age'])
		EvEvicted = int(request.form['EverEvicted'])
		CrimChargers = int(request.form['CriminalChargers'])
		FLBankruptcy = int(request.form['FileBankruptcy'])
		sec8 = int(request.form['IsSec8'])
		HsCollections = int(request.form['HasCollections'])
      
        
        data = np.array([[gen, mgi, fic, agev, EvEvicted, CrimChargers, FLBankruptcy, sec8, HsCollections]])
        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)