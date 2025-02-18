from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the saved scaler and stacking ensemble model
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("stacking_model.pkl", "rb") as f:
    stacking_model = pickle.load(f)

def compute_actual_grade(prev_grade, study_hours, attendance, absences, engagement, social_media):
    """
    Compute the actual final grade using the original formula:
    Final_Grade = 0.4*Previous_Grades + 0.2*Study_Hours_Per_Week + 0.15*Attendance_Percentage
                  - 0.1*Number_of_Absences + 0.1*Engagement_Score - 0.05*(Social_Media_Usage*5)
    The grade is clipped between 50 and 100.
    """
    grade = (0.4 * prev_grade +
             0.2 * study_hours +
             0.15 * attendance -
             0.1 * absences +
             0.1 * engagement -
             0.05 * social_media * 5)
    return int(np.clip(grade, 50, 100))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Retrieve input values from the form
            prev_grade = float(request.form.get("previous_grades"))
            study_hours = float(request.form.get("study_hours"))
            attendance = float(request.form.get("attendance"))
            absences = float(request.form.get("absences"))
            engagement = float(request.form.get("engagement"))
            social_media = float(request.form.get("social_media"))
            
            # Create DataFrame from the user input
            input_data = pd.DataFrame([{
                "Previous_Grades": prev_grade,
                "Study_Hours_Per_Week": study_hours,
                "Attendance_Percentage": attendance,
                "Number_of_Absences": absences,
                "Engagement_Score": engagement,
                "Social_Media_Usage": social_media
            }])
            
            # Standardize the input
            input_scaled = scaler.transform(input_data)
            
            # Perform prediction using the stacking ensemble model
            predicted_grade = stacking_model.predict(input_scaled)[0]
            
            # Compute the actual grade using the formula
            actual_grade = compute_actual_grade(prev_grade, study_hours, attendance, absences, engagement, social_media)
            
            # Calculate absolute error and check hypothesis with a threshold (e.g., ±5 points)
            abs_error = abs(actual_grade - predicted_grade)
            threshold = 5
            hypothesis = "PASS" if abs_error <= threshold else "FAIL"
            
            # Prepare results to display on the dashboard
            result = {
                "actual_grade": actual_grade,
                "predicted_grade": round(predicted_grade, 2),
                "absolute_error": round(abs_error, 2),
                "hypothesis": hypothesis
            }
            
            return render_template("index.html", result=result)
        except Exception as e:
            return render_template("index.html", error=str(e))
    else:
        return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
