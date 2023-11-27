from flask import Flask, request, render_template
import dill

app = Flask(__name__, template_folder = 'template')

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = dill.load(file)

#mapping    
gender_mapping = {'M': 0, 'F': 1}
car_mapping = {'Y': 0, 'N': 1}
realty_mapping = {'Y': 0, 'N': 1}
education_mapping = {
    'Higher education': 0,
    'Secondary / secondary special': 1,
    'Incomplete higher': 2,
    'Lower secondary': 3,
    'Academic degree': 4
}
family_status_mapping = {
    'Civil marriage': 0,
    'Married': 1,
    'Single / not married': 2,
    'Separated': 3,
    'Widow': 4
}
income_type_mapping = {
    'Working': 0,
    'Commercial associate': 1,
    'State servant': 2,
    'Pensioner': 3,
    'Student': 4
}
housing_mapping = {
    'Rented apartment': 0,
    'House / apartment': 1,
    'Municipal apartment': 2,
    'With parents': 3,
    'Co-op apartment': 4,
    'Office apartment': 5
}

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the HTML form
    features = {
        'GENDER': gender_mapping[request.form['gender']],
        'OWN_CAR': car_mapping[request.form['own_car']],
        'OWN_REALTY': realty_mapping[request.form['own_realty']],
        'AMT_INCOME_TOTAL': float(request.form['amt_income_total']),
        'INCOME_TYPE': income_type_mapping[request.form['income_type']],
        'EDUCATION_TYPE': education_mapping[request.form['education_type']],
        'FAMILY_STATUS': family_status_mapping[request.form['family_status']],
        'HOUSING_TYPE': housing_mapping[request.form['housing_type']],
        'AGE': -float(request.form['age']),
        'DAYS_EMPLOYED': float(request.form['days_employed']),
        'CNT_FAM_MEMBERS': float(request.form['cnt_fam_members'])
    }
    
    # Make a prediction using the loaded model
    #prediction = model.predict([features])[0]
    prediction = model.predict([list(features.values())])[0]

    
    # Return the prediction result to the HTML page
    return render_template('main.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
