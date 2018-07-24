# import all dependencies
from flask import Flask, request
import numpy as np
from sklearn.externals import joblib
import spacy
from flask import jsonify
app = Flask(__name__)


# load pertained models
nlp = spacy.load('en_core_web_md')
model = joblib.load("models/model")


# set up flask server
@app.route('/predict')
def query_example():
    # get all the values from the get request
    amount_pence = float(request.args.get('amount_pence'))
    balance_mortgage = float(request.args.get('balance_mortgage'))
    direct_debit_day = float(request.args.get('direct_debit_day'))
    estatus = float(request.args.get('estatus'))
    gross_income = float(request.args.get('gross_income'))
    home_value = float(request.args.get('home_value'))
    hstatus = float(request.args.get('hstatus'))
    monthly_mortgage = float(request.args.get('monthly_mortgage'))
    monthly_payment = float(request.args.get('monthly_payment'))
    net_income = float(request.args.get('net_income'))
    number_of_dependants = float(request.args.get('number_of_dependants'))
    reason = float(request.args.get('reason'))
    rent = float(request.args.get('rent'))
    work_number = float(request.args.get('work_number'))
    reasonother = nlp(str(request.args.get('reasonother'))).vector
    rsaddress_town = nlp(str(request.args.get('rsaddress_town'))).vector
    rsemployment_employers_name = nlp(str(request.args.get('rsemployment_employers_name'))).vector
    rsemployment_job_title = nlp(str(request.args.get('rsemployment_job_title'))).vector
    rsemployment_rsaddress_town = nlp(str(request.args.get('rsemployment_rsaddress_town'))).vector
    rsemployment_pre_employer_name = nlp(str(request.args.get('rsemployment_pre_employer_name'))).vector
    rsemployment_pre_job_title = nlp(str(request.args.get('rsemployment_pre_job_title'))).vector

    # convert vector values to avg
    reasonother = sum(reasonother) / len(reasonother)
    rsaddress_town = sum(rsaddress_town) / len(rsaddress_town)
    rsemployment_employers_name = sum(rsemployment_employers_name) / len(rsemployment_employers_name)
    rsemployment_job_title = sum(rsemployment_job_title) / len(rsemployment_job_title)
    rsemployment_rsaddress_town = sum(rsemployment_rsaddress_town) / len(rsemployment_rsaddress_town)
    rsemployment_pre_employer_name = sum(rsemployment_pre_employer_name) / len(rsemployment_pre_employer_name)
    rsemployment_pre_job_title = sum(rsemployment_pre_job_title) / len(rsemployment_pre_job_title)

    # load values into a np array
    model_values = np.array([amount_pence, balance_mortgage, direct_debit_day, estatus,
                             gross_income, home_value, hstatus, monthly_mortgage,
                             monthly_payment, net_income, number_of_dependants, reason,
                             reasonother, rent, rsaddress_town, rsemployment_employers_name,
                             rsemployment_job_title, rsemployment_pre_employer_name,
                             rsemployment_pre_job_title, rsemployment_rsaddress_town,
                             work_number]).astype(float)

    # get prediction probablities
    pred = model.predict_proba([model_values])[0]
    prediction = {'Percentage of approval': pred[1], 'percentage of rejection': pred[0]}
    return jsonify(prediction)


if __name__ == '__main__':
    app.run(debug=True, port=5000)  # run app in debug mode on port 5000
