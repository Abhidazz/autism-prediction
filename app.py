import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Load model and encoders
with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)
for col, le in encoders.items():
    print(col, le.classes_)


# ---- Helper function ----
def apply_encoders(df: pd.DataFrame, encoders_dict: dict) -> pd.DataFrame:
    df2 = df.copy()
    if encoders_dict is None:
        return df2
    

    for col, enc in encoders_dict.items():
        if col not in df2.columns:
            print(encoders)
            continue
        try:
            # If LabelEncoder
            if hasattr(enc, "classes_"):
                df2[col] = enc.transform(df2[col].astype(str))
            # If OneHotEncoder
            elif hasattr(enc, "categories_"):
                arr = enc.transform(df2[[col]]).toarray()
                names = enc.get_feature_names_out([col])
                for i, name in enumerate(names):
                    df2[name] = arr[:, i]
                df2 = df2.drop(columns=[col])
        except Exception as e:
            print(f"Encoding failed for {col}: {e}")

    # Align with model's expected input features
    if hasattr(model, 'feature_names_in_'):
        expected = list(model.feature_names_in_)
        for c in expected:
            if c not in df2.columns:
                df2[c] = 0
        df2 = df2[expected]

    return df2

# ---- Routes ----
@app.route('/')
def index():
    # this loads templates/index.html
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convert form to dataframe
        form_data = request.form.to_dict()
        df = pd.DataFrame([form_data])

        mappings = {
    "gender": {"male": "m", "female": "f"},
    "jaundice": {"yes": "yes", "no": "no"},
    "austim": {"yes": "yes", "no": "no"},
    "used_app_before": {"yes": "yes", "no": "no"},
}

        # Apply mappings first
        for col, mapping in mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)

        # Now apply LabelEncoders
        for col, le in encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col].astype(str))


       
        # Make sure numeric columns are converted properly
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass

        # Align with model's expected input features
        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
            for c in expected:
                if c not in df.columns:
                    df[c] = 0 
            df = df[expected]

        # Predict
        prediction = model.predict(df)[0]
        
        prediction_text = "Has Autism" if prediction == 1 else "No Autism"
        return render_template("results.html", prediction=prediction_text)

    except Exception as e:
        return f"Error during prediction: {e}"

if __name__ == '__main__':
    app.run(debug=True)
