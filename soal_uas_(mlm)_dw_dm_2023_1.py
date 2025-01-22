# -*- coding: utf-8 -*-
"""Soal UAS (Mlm) - DW-DM - 2023-1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1H3lll4nAQQHztcXl2dQprZ2VVaA0q5WW

# <center>UAS (Malam) : Data Warehouse dan Data Mining</center>

- NIM : 2021230039
- Nama : Ryno Pahlevi Al Ghiffari

---

# <u>Soal :</u>
## <font color=red>Berikut Program membangun model regresi (prediksi pembayaran premi asuransi), koreksi dan lengkapi jika ada yang perlu diperbaiki, simpan hasil/model-nya dan gunakan untuk membangun aplikasi web  dengan Streamlit, upload ke server streamlit !</font>

---
"""

import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

"""### Loading Dataset"""

df = pd.read_csv('insurance1.csv')
df.head()

df.info()

df.describe()

df.isna().sum()

"""### Data Preparation - Mengubah data categorical ke numeric
Perbaiki jika ada ERROR
"""

from sklearn.preprocessing import OrdinalEncoder
oe=OrdinalEncoder()
# df[['sex','smoker','region']]=oe.fit_transform(df[['sex','smoker','region']])
df[['sex','smoker']]=oe.fit_transform(df[['sex','smoker']])

# df[['sex','smoker','region']]=df[['sex','smoker','region']].astype(int)
df[['sex','smoker']]=df[['sex','smoker']].astype(int)

df.head()

df.columns

"""### Memisah Feature dan Label"""

target=df['charges']
features=df.drop(columns=['charges'])

#Standardization
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
features=sc.fit_transform(features)

pd.DataFrame(features[:5])

"""### Membangun Model"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x_train,x_test,y_train,y_test=train_test_split(features,target,shuffle=True,train_size=0.8,random_state=1)

len(x_train)

len(x_test)

#Training
model_uas=LinearRegression()
model_uas.fit(x_train,y_train)

"""### Evaluasi"""

from sklearn.metrics import r2_score,mean_squared_error
pred=model_uas.predict(x_test)
print('r2 score :',r2_score(y_test,pred))
print('MSE :',mean_squared_error(y_test,pred))

print(pred[:5])
print(y_test[:5])

"""### Menyimpan Model untuk Deployment

Install pickle jika belum ada

  !pip install pickle-mixin
"""

import pickle

pickle.dump(model_uas,open('model_uas.pkl','wb'))

"""Cek file model_uas.pkl, dan gunakan sebagai model saat membangun aplikasi web berbasis streamlit

---
### Clue
Untuk membangun model ini ke aplikasi web streamlit :
- Model terdiri dari 5 atribut input dan 1 output :

  INPUT :
  - 'age', umur
  - 'sex',  jenis kelamin
  - 'bmi',  berat
  - 'children',  jumlah anak
  - 'smoker',  perokok atau tidak
  
  OUTPUT
  - charges, besar iuran asuransi per bulan
"""

#Contoh penggunaan
#Yang diinputkan
age=27
sex=1
bmi=30
children=5
smoker=0

#proses input - dijadikan array dan di reshape
X=np.array([age,sex, bmi, children, smoker])
X=X.reshape(1,-1)

#load model tersimpan - pickle
loaded_model = pickle.load(open('model_uas.pkl', 'rb'))

#prediksi
charger_pred=loaded_model.predict(X)

#hasil setelah disubmit
print("Prediksi Pembayaran premi : ",charger_pred)

"""---"""