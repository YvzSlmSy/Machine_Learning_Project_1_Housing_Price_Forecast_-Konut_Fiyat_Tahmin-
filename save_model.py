import os
import sys

# Proje kök dizinini ekleyelim
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import joblib
from src.train import train_model, evaluate_model
from src.load_data import load_housing_data
from src.preprocess import preprocess_data

# Modeli eğit ve değerlendir
df = load_housing_data()
X_train, X_test, y_train, y_test = preprocess_data(df)

model = train_model(X_train, y_train)
evaluate_model(model, X_test, y_test)

# Modeli kaydet
joblib.dump(model, 'models/house_price_model.pkl')
print("Model başarıyla kaydedildi.")
