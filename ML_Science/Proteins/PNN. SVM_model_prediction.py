# Cargar bibliotecas de Python
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
import pandas as pd
import numpy as np

# Cargar modelo entrenado
filename = "path_to_the_saved_model" # The extension needs to be .SV
idr_model = joblib.load(filename)

# Cargar datos de la nueva IDR a predecir
df_IDRs = pd.read_csv("path_of_your_data") # Use a CSV file

# Obtener los datos numéricos para entrenar el modelo SVM
pred_IDR = df_IDRs.iloc[:,[2, 3, 4]].values

# Defninir un "scaler"
scaler = StandardScaler()

# Escalar los datos de prueba
pred_IDR = scaler.fit_transform(pred_IDR)

# Realizar predicción de la IDR
pred_result = idr_model.predict(pred_IDR)

# Convertir resultados en data frame
df_pred = pd.DataFrame(pred_result.tolist(), columns = ["Prediction"])

# Juntar data frames
df_IDRs.insert(len(df_IDRs.columns), "Prediction", df_pred)

# Guardar data frame
df_IDRs.to_csv("path_for_saving_your_predicted_data") 
