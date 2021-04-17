from flask import Flask, render_template, request
import pickle
import numpy as np

filename='earthproject.pkl'
dt=pickle.load(open(filename,'rb'))

app=Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def man():
	if request.method=='POST':
		age=int(request.form['age'])
		cf=int(request.form['count_floors_pre_eq'])
		ap=int(request.form['area_percentage'])
		hp=int(request.form['height_percentage'])
		lsc=int(request.form['land_surface_condition'])
		ft=int(request.form['foundation_type'])
		hsuh=int(request.form['has_secondary_use_hotel'])
		hsur=int(request.form['has_secondary_use_rental'])
		rf=int(request.form['roof_type'])

		arr=np.array([[age,cf,ap,hp,lsc,ft,hsuh,hsur,rf]])
		pred=dt.predict(arr)

		return render_template('after.html',data=pred)

if __name__ == "__main__":
	app.run(debug=True)
