import pickle
import gradio as gr

"""
Bank Loan Prediction App (Inference Only)

This Gradio application loads a pre-trained machine learning model and scaler 
to predict whether a bank customer will accept a personal loan. 
If the model or scaler cannot be loaded, the app will still launch and display 
an error message in the prediction output.
"""

model = None
scaler = None
load_error = None

try:
    with open("loan_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    load_error = f"❌ Error loading model or scaler: {e}"


def predict_loan(Age, Experience, Income, Family, CCAvg, Education, Mortgage, Securities, CD, Online, CreditCard):
    """
    Predict whether a customer will accept a personal loan.

    Parameters:
        Age (float): Age of the customer
        Experience (float): Years of professional experience
        Income (float): Annual income in $1000s
        Family (float): Number of family members
        CCAvg (float): Average monthly credit card spending in $1000s
        Education (int): Education level (1=Undergrad, 2=Graduate, 3=Advanced/Professional)
        Mortgage (float): Value of house mortgage if any
        Securities (bool): Whether the customer has a securities account
        CD (bool): Whether the customer has a certificate of deposit (CD) account
        Online (bool): Whether the customer uses online banking
        CreditCard (bool): Whether the customer has a credit card with the bank

    Returns:
        dict: A dictionary containing either:
              - "Prediction" and "Probability" if successful
              - "Error" if an error occurs or the model is unavailable
    """
    if load_error:
        return {"Error": load_error}

    try:
        input_data = [[Age, Experience, Income, Family, CCAvg, Education, Mortgage,
                       int(Securities), int(CD), int(Online), int(CreditCard)]]
        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]
        return {"Prediction": "✅ Will Take Loan" if pred == 1 else "❌ Will Not Take Loan",
                "Probability": f"{prob:.2f}"}
    except Exception as e:
        return {"Error": str(e)}


inputs = [
    gr.Number(label="Age"),
    gr.Number(label="Experience"),
    gr.Number(label="Income (in $1000s)"),
    gr.Number(label="Family Members"),
    gr.Number(label="CCAvg (Monthly credit card spending in $1000s)"),
    gr.Dropdown([1, 2, 3], label="Education (1=Undergrad, 2=Graduate, 3=Advanced)"),
    gr.Number(label="Mortgage"),
    gr.Checkbox(label="Securities Account"),
    gr.Checkbox(label="CD Account"),
    gr.Checkbox(label="Online Banking"),
    gr.Checkbox(label="Credit Card"),
]

outputs = gr.JSON()

app = gr.Interface(
    fn=predict_loan,
    inputs=inputs,
    outputs=outputs,
    title="Bank Loan Prediction (Inference Only)"
)

app.launch()