import numpy as np
import pandas as pd 
import sklearn
from flask import Flask,request,jsonify,render_template
import pickle

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))


dataset=pd.read_csv('diabetes-pima.csv')


dataset_X=dataset.iloc[:,[1,5,7,4,2,3,0,6]].values


from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
dataset_scaled=sc.fit_transform(dataset_X)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict():
    float_features=[float(x) for x in request.form.values()]
    final_features=[np.array(float_features)]
    prediction=model.predict(sc.transform(final_features))

    if  prediction==1:
        pred="You have Diabetes and it may be type-2, please consult a doctor."
    else :
        pred="You Don't have Diabetes."
    out=pred 
    return render_template('index.html',prediction_text='{}'.format(out))

if __name__=="__main__":
    app.run(debug=True)