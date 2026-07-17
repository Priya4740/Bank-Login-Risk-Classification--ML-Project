from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Paths
MODEL_PATH = "/home/MaruboinaPriyanka/mysite/threat_detection_model.pkl"
SCALER_PATH = "/home/MaruboinaPriyanka/mysite/scaler.pkl"

# Load model and scaler
classifier = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Reduce XGBoost threads
try:
    classifier.set_params(n_jobs=1)
except:
    pass


@app.route("/", methods=["GET", "POST"])
def predict():

    prediction = ""

    if request.method == "POST":
        try:
            login_hour = int(request.form["login_hour"])
            device_type = int(request.form["device_type"])
            browser = int(request.form["browser"])
            failure_reason = int(request.form["failure_reason"])
            auth_type = int(request.form["auth_type"])
            ip_category = int(request.form["ip_category"])

            # Feature order must match training
            new_data = pd.DataFrame({
                "device_type": [device_type],
                "browser": [browser],
                "failure_reason": [failure_reason],
                "auth_type": [auth_type],
                "login_hour": [login_hour],
                "ip_category": [ip_category]
            })

            print("Input Data:")
            print(new_data)

            # Scale input
            new_data_scaled = scaler.transform(new_data)

            # Predict
            pred = classifier.predict(new_data_scaled)

            print("Predicted Class:", pred[0])

            # Convert class into Low/Medium/High
            if pred[0] <= 3:
                prediction = f"🟢 Low Threat Login"
            elif pred[0] <= 7:
                prediction = f"🟡 Medium Threat Login"
            else:
                prediction = f"🔴 High Threat Login"

        except Exception as e:
            prediction = f"Error: {e}"
            print(e)

        # Show result on a new page
        return render_template("result.html", prediction=prediction)

    # Show the input form
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)