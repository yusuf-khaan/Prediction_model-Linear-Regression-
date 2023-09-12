import pandas as pd
import pickle

# Load the trained model
with open('predicting_model_subsetfinal.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def predict_quantity(customer_id, day_of_month, month, year):
    # DataFrame with the input data
    input_data = pd.DataFrame({
        'Year': [year],                         
        'Date': [day_of_month],                     # Replace with the actual column name
        'Month': [month],                       # Replace with the actual column name
        'CustomerID': [customer_id]                # Replace with the actual column name
    })

    # Make predictions using the trained model
    predicted_quantity = model.predict(input_data)  # based on preprocessing steps

    return predicted_quantity[0]  # Assuming a single prediction is made

# Take user input for customer_id, day_of_month, month, and year
customer_id = int(input("Enter Customer ID: "))
day_of_month = int(input("Enter Day of the Month: "))
month = int(input("Enter Month: "))
year = int(input("Enter Year: "))

predicted_quantity = predict_quantity(customer_id, day_of_month, month, year)
print(f"Predicted Quantity for Customer ID {customer_id} on {year}-{month:02d}-{day_of_month:02d}: {predicted_quantity}")
