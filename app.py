from flask import Flask, render_template, request
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd


app = Flask(__name__)

# تحميل النموذج المدرب مسبقاً (افترض أنك قمت بتدريب النموذج وحفظته باستخدام joblib)
svm_clf=joblib.load('/Users/shahaalfughom/Desktop/Heart Disease App/heart_disease_model.pkl')
scaler = joblib.load('/Users/shahaalfughom/Desktop/Heart Disease App/scaler.pkl')  # إذا كنت تستخدم StandardScaler أو أي محول آخر

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # الحصول على البيانات المدخلة من المستخدم عبر الفورم
        age = float(request.form['Age'])
        restingbp = float(request.form['RestingBP'])
        cholesterol = float(request.form['Cholesterol'])
        maxhr = float(request.form['MaxHR'])
        oldpeak = float(request.form['Oldpeak'])

        # القيم الفئوية يتم الحصول عليها من خلال select
        sex = int(request.form['Sex'])  # 1 للذكر و 0 للأنثى
        chestpain = int(request.form['ChestPainType'])  # 1, 2, 3, 4
        fastingbs = int(request.form['FastingBS'])  # 0 أو 1
        restingecg = int(request.form['RestingECG'])  # 0, 1, 2
        exerciseangina = int(request.form['ExerciseAngina'])  # 1 أو 0
        st_slope = int(request.form['ST_Slope'])  # 1, 2, 3

        # إنشاء DataFrame للبيانات المدخلة
        input_data = pd.DataFrame([[age, restingbp, cholesterol, maxhr, oldpeak, sex, chestpain, fastingbs, restingecg,
                                    exerciseangina, st_slope]],
                                  columns=['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak',
                                           'Sex', 'ChestPainType', 'FastingBS', 'RestingECG',
                                           'ExerciseAngina', 'ST_Slope'])

        # تحويل المتغيرات الفئوية إلى متغيرات وهمية (One-Hot Encoding)
        categorical_columns = ['ChestPainType', 'RestingECG', 'ST_Slope']
        input_data = pd.get_dummies(input_data, columns=categorical_columns, drop_first=True)

        # إضافة الأعمدة الناقصة وضمان ترتيب الأعمدة بنفس شكل بيانات التدريب
        for col in svm_clf.feature_names_in_:
            if col not in input_data.columns:
                input_data[col] = 0  # تعويض الأعمدة الناقصة بالقيمة 0
        input_data = input_data[svm_clf.feature_names_in_]  # إعادة الترتيب

        # تحجيم الأعمدة الرقمية
        numeric_columns = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
        input_data[numeric_columns] = scaler.transform(input_data[numeric_columns])

        # إجراء التنبؤ
        prediction = svm_clf.predict(input_data)
        result = 'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'

        return render_template('result.html', prediction_text=f"Prediction: {result}")

    except Exception as e:
        return render_template('result.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
       app.run(debug=True)