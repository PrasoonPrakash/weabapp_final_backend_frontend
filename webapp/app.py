from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from flask_caching import Cache
import numpy as np
import pandas as pd
import catboost
from datetime import datetime
import json, os

# Storing request data here.
STORAGE_BASE = './data/'
os.makedirs(STORAGE_BASE, exist_ok=True)

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Load the scaler
with open('./model/scale.pkl','rb') as f:
    preprocessor = pickle.load(f)

# Load the trained model
with open('./model/model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/')
@cache.cached(timeout=50)
def home():
    return render_template('index.html')

@app.route('/one(6)')
@cache.cached(timeout=50)
def one6():
    return render_template('one(6).html')

@app.route('/test-centres')
@cache.cached(timeout=50)
def testcentres():
    return render_template('test-centres.html')

@app.route('/about-us')
@cache.cached(timeout=50)
def aboutus():
    return render_template('about-us.html')

@app.route('/test-centress')
@cache.cached(timeout=50)
def testcentress():
    return render_template('test-centress.html')

@app.route('/one')
@cache.cached(timeout=50)
def one():
    return render_template('one.html')

@app.route('/two')
@cache.cached(timeout=50)
def two():
    return render_template('two.html')

@app.route('/three')
@cache.cached(timeout=50)
def three():
    return render_template('three.html')

@app.route('/four')
@cache.cached(timeout=50)
def four():
    return render_template('four.html')


def generate_request_id():
    # Get the current date and time
    now = datetime.now()
    # Format it as "yyyy-mm-dd-hh-mm-ss"
    request_id = now.strftime("%Y-%m-%d-%H-%M-%S")
    return request_id


def create_dummy_variables(user_input):
    features = [
        'AGE',
        'AGE_FIRST CHILD BIRTH',
        'CONTRACEPTIVE_DURATION_MNTHS',
        'PHYSICAL ACTIVITY_DURATION_MINS',
        'HEIGHT (IN CMS)',
        'EDUCATION',
        'OCCUPATION',
        'FAMILY TYPE',
        'REG_MENSTRUATION_History',
        'MENSTRUAL_STATUS',
        'HOT FLUSHES',
        'CONCEPTION(NATURALLY CONCEIVE/IVF)',
        'HRT',
        'FAMILYHO_CANCER',
        'FAMILYHO_MEMBER_TYPE',
        'RTI/STI',
        'FASTING',
        'MUSTARD OIL',
        'BREAST TRAUMA',
        'RADIATION_SITE_CHEST'
    ]
                  
    input_data = dict()
    for i in features:
      input_data[i] = user_input.get(i)
      
    return input_data


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            TAG = generate_request_id()
            print('A new request recevied. TAG:', TAG)

            features =  [
                'AGE',
                'AGE_FIRST CHILD BIRTH',
                'CONTRACEPTIVE_DURATION_MNTHS',
                'PHYSICAL ACTIVITY_DURATION_MINS',
                'HEIGHT (IN CMS)',
                'EDUCATION',
                'OCCUPATION',
                'FAMILY TYPE',
                'REG_MENSTRUATION_History',
                'MENSTRUAL_STATUS',
                'HOT FLUSHES',
                'CONCEPTION(NATURALLY CONCEIVE/IVF)',
                'HRT',
                'FAMILYHO_CANCER',
                'FAMILYHO_MEMBER_TYPE',
                'RTI/STI',
                'FASTING',
                'MUSTARD OIL',
                'BREAST TRAUMA',
                'RADIATION_SITE_CHEST'
            ]

            numerical_cols = [
                "AGE", "AGE_FIRST CHILD BIRTH",
                "CONTRACEPTIVE_DURATION_MNTHS",
                "PHYSICAL ACTIVITY_DURATION_MINS",
                "HEIGHT (IN CMS)"
            ]

            user_input = dict()
            for feat in features:
                user_input[feat] = request.form[feat]
                if feat in numerical_cols:
                    user_input[feat] = float(user_input[feat])
            input_data = create_dummy_variables(user_input)

            # Save the collected data.
            with open(os.path.join(STORAGE_BASE, f"{TAG}.json"), 'w') as fp:
                json.dump(input_data, fp)
            
            # Optionally, convert input_data to DataFrame or use it directly for prediction
            input_df = pd.DataFrame([input_data])

            df = preprocessor.transform(input_df)
            df = [j for elem in df for j in elem]
            df = dict(zip(features, df))
            df = pd.DataFrame(df,index=[0])

            # df.to_csv('processed.csv', index=False)
            input_df = df.rename(columns={
                'AGE': 'num__AGE', 
                'AGE_FIRST CHILD BIRTH':'num__AGE_FIRST CHILD BIRTH',
                'CONTRACEPTIVE_DURATION_MNTHS':'num__CONTRACEPTIVE_DURATION_MNTHS',
                'PHYSICAL ACTIVITY_DURATION_MINS':'num__PHYSICAL ACTIVITY_DURATION_MINS',
                'HEIGHT (IN CMS)':'num__HEIGHT (IN CMS)',
                'EDUCATION':'cat__EDUCATION',
                'OCCUPATION':'cat__OCCUPATION',
                'FAMILY TYPE':'cat__FAMILY TYPE',
                'REG_MENSTRUATION_History':'cat__REG_MENSTRUATION_History',
                'MENSTRUAL_STATUS':'cat__MENSTRUAL_STATUS',
                'HOT FLUSHES':'cat__HOT FLUSHES',
                'CONCEPTION(NATURALLY CONCEIVE/IVF)':'cat__CONCEPTION(NATURALLY CONCEIVE/IVF)',
                'HRT':'cat__HRT',
                'FAMILYHO_CANCER':'cat__FAMILYHO_CANCER',
                'FAMILYHO_MEMBER_TYPE':'cat__FAMILYHO_MEMBER_TYPE',
                'RTI/STI':'cat__RTI/STI',
                'FASTING':'cat__FASTING',
                'MUSTARD OIL':'cat__MUSTARD OIL',
                'BREAST TRAUMA':'cat__BREAST TRAUMA',
                'RADIATION_SITE_CHEST':'cat__RADIATION_SITE_CHEST'
            })
            
            # Make predictions using the loaded model
            prediction = model.predict(input_df)
            probability_scores = model.predict_proba(input_df)
            probability_scores = [j for elem in probability_scores for j in elem]
            for i in range(len(probability_scores)):
                if i == 0:
                    prob_0 = probability_scores[i]
                    prob_0 = round((prob_0 * 100), 2)
                else:
                    prob_1 = probability_scores[i]
                    prob_1 = round((prob_1 * 100), 2)

            if prediction == 1:
                message = f"Based on your assessment, there is an estimated {prob_1}% risk of breast cancer.\n\n\n\n To ensure your health and well-being, we will schedule an appointment for you at the earliest possible date."
                message_hindi = f"आपके मूल्यांकन के आधार पर, स्तन कैंसर का {prob_1}% अनुमानित खतरा है।\n\n\n\n आपके स्वास्थ्य और भले के लिए, हम आपके लिए संभाविततम तारीख पर नियुक्ति निर्धारित करेंगे."
            elif prediction == 0:
                message = f"Based on your assessment, there is an estimated {prob_0}% risk of NOT having breast cancer.\n\n\n\nYou can continue with your regular followup with your physician."
                message_hindi = f"आपके मूल्यांकन के आधार पर, स्तन कैंसर न होने की संभावना {prob_0}% है।\n\n\n\n आप अपने नियमित फॉलो-अप को अपने चिकित्सक के साथ जारी रख सकते हैं।"
                
            return render_template('result(6).html', result=message, result_hindi=message_hindi)

        except Exception as e:
            return str(e)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
