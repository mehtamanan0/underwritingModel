# Underwriting Model

## Installation

**Disturbution Tested on**
- Ubuntu server 16.04

**Requirments**

- python3

#### Install pip3

```
sudo apt install python3-pip
```

#### Clone the repository

```
git clone git@gl.commuterclub.co.uk:main/cc_underwritingModel.git; cd cc_underwritingModel; git checkout model
```

#### Install python dependencies

```
pip3 install -r requirements.txt
python3 -m spacy download en_core_web_md
```

## Running the server

#### Start the server

```
./start_server.sh
```

Once the server is running all the log will be saved to log.txt

#### Stop the server

```
./stop_server.sh
```

## Testing

**NOTE: change the host address to the address
on which the server is running**

#### Using a browser

URL :
```
http://127.0.0.1:5000/predict?amount_pence=113542&balance_mortgage=0&direct_debit_day=2&estatus=8&gross_income=0&home_value=0&hstatus=1&monthly_mortgage=0&monthly_payment=120.58&net_income=0&number_of_dependants=0&reason=10&reasonother=CC-TravelTicket&rent=0&rsaddress_town=London&rsemployment_employers_name=0&rsemployment_job_title=0&rsemployment_pre_employer_name=0&rsemployment_pre_job_title=0&rsemployment_rsaddress_town=0&work_number=7886870267
```

Copy the link and paste it into a browser

#### Using curl

```
curl "http://127.0.0.1:5000/predict?amount_pence=113542&balance_mortgage=0&direct_debit_day=2&estatus=8&gross_income=0&home_value=0&hstatus=1&monthly_mortgage=0&monthly_payment=120.58&net_income=0&number_of_dependants=0&reason=10&reasonother=CC-TravelTicket&rent=0&rsaddress_town=London&rsemployment_employers_name=0&rsemployment_job_title=0&rsemployment_pre_employer_name=0&rsemployment_pre_job_title=0&rsemployment_rsaddress_town=0&work_number=7886870267"
```

#### The output should look something like this :

```
{
  "Percentage of approval": 0.6109171426229283,
  "percentage of rejection": 0.3890828573770717
}
```

## File Functions and Explanations

### predict_server.py

#### Function

Creates a REST api server for GET requests. It can be accessed by giving it the Api parameters given below.
Once it gets the parameters, it will give these parameters to the model which will then predict probabilities
and return in a json format.

#### Explanations

this file will load the model from the `models/` directory as well as the vector model here :

```
nlp = spacy.load('en_core_web_md')
model = joblib.load("models/model")
```

Then the the function `predict` will fetch GET request values like so:

```
amount_pence = float(request.args.get('amount_pence'))
```

After getting all parameters, all the parameters will be sent to the model for probability prediction:

```
pred = model.predict_proba([model_values])[0]
```

After that the request will return the output in a json format

### train_model.py

#### Function

This file reads the preprocessed data and generates a model that gets saved in the `models` directory

#### Explanations

The file will first read the preprocessed csv file:
```
train_df = pd.read_csv("data/train_data_preprocessed.csv")
```

Then the training data will be split like:

```
X = train_df.values[:, 0:21].astype(float)

Y = train_df.values[:, 21].astype(int)
```

Here **X** contains all the data values for training
And **Y** contains the corresponding labels for the data where:

- 1 - Approved
- 0 - Declined

Then the model will be trained like so:

```
model.fit(X, Y)
```

After training the model the model will be saved in the models directory

## Api arguments

The api is a GET request with the following parameters:

- amount_pence **(int)**
- balance_mortgage **(int)**
- direct_debit_day **(int)**
- estatus **(int)**
- gross_income **(int)**
- home_value **(int)**
- hstatus **(int)**
- monthly_mortgage **(int)**
- monthly_payment **(int)**
- net_income **(int)**
- number_of_dependants **(int)**
- reason **(int)**
- rent **(int)**
- work_number **(int)**
- reasonother **(string)**
- rsaddress_town **(string)**
- rsemployment_employers_name **(string)**
- rsemployment_job_title **(string)**
- rsemployment_rsaddress_town **(string)**
- rsemployment_pre_employer_name **(string)**
- rsemployment_pre_job_title **(string)**

## NOTE

##### if you get the following error:

```
Traceback (most recent call last):
  File "predict_server.py", line 11, in <module>
    model = joblib.load("models/model")
  File "/home/manan/.local/lib/python3.5/site-packages/sklearn/externals/joblib/numpy_pickle.py", line 578, in load
    obj = _unpickle(fobj, filename, mmap_mode)
  File "/home/manan/.local/lib/python3.5/site-packages/sklearn/externals/joblib/numpy_pickle.py", line 508, in _unpickle
    obj = unpickler.load()
  File "/usr/lib/python3.5/pickle.py", line 1039, in load
    dispatch[key[0]](self)
  File "/home/manan/.local/lib/python3.5/site-packages/sklearn/externals/joblib/numpy_pickle.py", line 341, in load_build
    self.stack.append(array_wrapper.read(self))
  File "/home/manan/.local/lib/python3.5/site-packages/sklearn/externals/joblib/numpy_pickle.py", line 184, in read
    array = self.read_array(unpickler)
  File "/home/manan/.local/lib/python3.5/site-packages/sklearn/externals/joblib/numpy_pickle.py", line 108, in read_array
    array = pickle.load(unpickler.file_handle)
  File "sklearn/tree/_tree.pyx", line 601, in sklearn.tree._tree.Tree.__cinit__
ValueError: Buffer dtype mismatch, expected 'SIZE_t' but got 'long long'
```

#### Run

```
python3 train_model.py
```

This is caused due to the model mismatch on different architectures.
The above command will recreate the model on current architecture.

