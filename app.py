import pickle
from flask import Flask, render_template, request, redirect, url_for
from flask_mysqldb import MySQL
from config import Config
import numpy as np

app = Flask(__name__)
app.config.from_object(Config)
mysql = MySQL(app)

# Load the trained attrition prediction model
with open('attrition_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler used during training
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Home route to display all employees
@app.route('/')
def index():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM employees")
    employees = cur.fetchall()
    cur.close()
    return render_template('index.html', employees=employees)

# Add Employee
@app.route('/add', methods=['GET', 'POST'])
def add_employee():
    if request.method == 'POST':
        data = request.form
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO employees (employee_id, age, gender, years_at_company, job_role, monthly_income, work_life_balance, job_satisfaction, performance_rating, number_of_promotions, overtime, distance_from_home, education_level, marital_status, number_of_dependents, job_level, company_size, company_tenure, remote_work, leadership_opportunities, innovation_opportunities, company_reputation, employee_recognition, attrition) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", 
                    (data['employee_id'], data['age'], data['gender'], data['years_at_company'], data['job_role'], data['monthly_income'], data['work_life_balance'], data['job_satisfaction'], data['performance_rating'], data['number_of_promotions'], data['overtime'], data['distance_from_home'], data['education_level'], data['marital_status'], data['number_of_dependents'], data['job_level'], data['company_size'], data['company_tenure'], data['remote_work'], data['leadership_opportunities'], data['innovation_opportunities'], data['company_reputation'], data['employee_recognition'], data['attrition']))
        mysql.connection.commit()
        cur.close()
        return redirect(url_for('index'))
    return render_template('add_employee.html')

# Edit Employee
@app.route('/edit/<int:employee_id>', methods=['GET', 'POST'])
def edit_employee(employee_id):
    cur = mysql.connection.cursor()
    if request.method == 'POST':
        data = request.form
        cur.execute("UPDATE employees SET age=%s, gender=%s, years_at_company=%s, job_role=%s, monthly_income=%s, work_life_balance=%s, job_satisfaction=%s, performance_rating=%s, number_of_promotions=%s, overtime=%s, distance_from_home=%s, education_level=%s, marital_status=%s, number_of_dependents=%s, job_level=%s, company_size=%s, company_tenure=%s, remote_work=%s, leadership_opportunities=%s, innovation_opportunities=%s, company_reputation=%s, employee_recognition=%s, attrition=%s WHERE employee_id=%s", 
                    (data['age'], data['gender'], data['years_at_company'], data['job_role'], data['monthly_income'], data['work_life_balance'], data['job_satisfaction'], data['performance_rating'], data['number_of_promotions'], data['overtime'], data['distance_from_home'], data['education_level'], data['marital_status'], data['number_of_dependents'], data['job_level'], data['company_size'], data['company_tenure'], data['remote_work'], data['leadership_opportunities'], data['innovation_opportunities'], data['company_reputation'], data['employee_recognition'], data['attrition'], employee_id))
        mysql.connection.commit()
        cur.close()
        return redirect(url_for('index'))
    cur.execute("SELECT * FROM employees WHERE employee_id=%s", (employee_id,))
    employee = cur.fetchone()
    cur.close()
    return render_template('edit_employee.html', employee=employee)

# Delete Employee
@app.route('/delete/<int:employee_id>')
def delete_employee(employee_id):
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM employees WHERE employee_id=%s", (employee_id,))
    mysql.connection.commit()
    cur.close()
    return redirect(url_for('index'))

# Predict Employee Attrition
@app.route('/predict_attrition', methods=['GET', 'POST'])
def predict_attrition():
    if request.method == 'POST':
        data = request.form
        # Prepare data for prediction (make sure the order matches your model training)
        input_data = np.array([[data['age'], data['years_at_company'], data['monthly_income'], 
                                data['work_life_balance'], data['job_satisfaction'], 
                                data['performance_rating'], data['number_of_promotions'], 
                                data['overtime'], data['distance_from_home'], 
                                data['education_level'], data['number_of_dependents'], 
                                data['job_level'], data['company_size'], data['company_tenure'],
                                data['remote_work'], data['leadership_opportunities'],
                                data['innovation_opportunities'], data['company_reputation'],
                                data['employee_recognition']]]).astype(float)

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Make prediction (0 = Stayed, 1 = Left)
        prediction = model.predict(input_data_scaled)
        result = 'Left' if prediction == 1 else 'Stayed'
        
        return render_template('result.html', prediction=result)
    return render_template('predict_attrition.html')

if __name__ == '__main__':
    app.run(debug=True)

