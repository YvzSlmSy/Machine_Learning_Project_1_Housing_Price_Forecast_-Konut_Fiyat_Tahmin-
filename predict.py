import os
import sys

# Proje kök dizinini ekleyelim
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import joblib
import numpy as np

# Kaydedilen modelin yüklenmesi
model = joblib.load('models/house_price_model.pkl')

# Örnek veriyle tahmin yapma
new_data = np.array([[5.0, 20.0, 6.0, 2.0, 2000.0, 3.0, 37.0, -122.0]])  # Yeni örnek veri
prediction = model.predict(new_data)
print(f"Tahmin Edilen Konut Fiyatı: {prediction[0]}")
