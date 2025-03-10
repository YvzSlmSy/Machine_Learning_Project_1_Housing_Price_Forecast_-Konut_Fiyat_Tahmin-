'''
verinin yüklenmesi ile eksik verilerin kontrol edilmesi
veri dağılımının icelenmesi
korelasyon matrisinin incelenmesi

'''
import os
import sys

# Proje kök dizinini ekleyelim
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.load_data import load_housing_data  # Veriyi yükleme fonksiyonunu içe aktar

# Veriyi yükleyelim
df = load_housing_data()

# Veri tipi ve eksik veri kontrolü
print(df.info())
print(df.isnull().sum())

# Veri dağılımını inceleyelim
df.hist(figsize=(12, 8), bins=30)
plt.show()

# Korelasyon matrisi
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Korelasyonları")
plt.show()
