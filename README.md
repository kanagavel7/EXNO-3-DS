## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
Developed by:KANAGAVEL.R
Regno:212223040085
```
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```

![Screenshot 2024-10-07 083856](https://github.com/user-attachments/assets/9a055db3-2efb-4445-89cf-1dc4c644e736)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

![Screenshot 2024-10-07 083945](https://github.com/user-attachments/assets/12dc7094-1dc8-4695-902d-3e4180693638)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

![Screenshot 2024-10-07 084022](https://github.com/user-attachments/assets/5d0d1212-03df-445f-b91e-182b397e4aa7)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```

![Screenshot 2024-10-07 084053](https://github.com/user-attachments/assets/de6941b1-b0ec-47ef-a059-0fcc1e26cf10)

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```
```
df2=pd.concat([df2,enc],axis=1)
df2
```

![Screenshot 2024-10-07 084205](https://github.com/user-attachments/assets/1515142a-8058-4b81-b058-a82994f372b2)

```
pd.get_dummies(df2,columns=["nom_0"])
```

![Screenshot 2024-10-07 084237](https://github.com/user-attachments/assets/e8000fc7-2374-46ba-a174-26e78a6816c2)

```
pip install --upgrade category_encoders
```
```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```
```
dfb=pd.concat([df,nd],axis=1)
dfb
```

![Screenshot 2024-10-07 084400](https://github.com/user-attachments/assets/9f34096b-6b52-436c-8b80-40d3135ed52e)

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```

![Screenshot 2024-10-07 084432](https://github.com/user-attachments/assets/629221f2-a899-4d49-97c9-9f9eddcc0ec8)

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```

![Screenshot 2024-10-07 084504](https://github.com/user-attachments/assets/fc9340fe-9230-442f-ae86-71d6b2727832)

```
df.skew()
```

![Screenshot 2024-10-07 084536](https://github.com/user-attachments/assets/24639423-9308-4286-9f77-35ec50e404af)

```
np.log(df["Highly Positive Skew"])
```

![Screenshot 2024-10-07 084615](https://github.com/user-attachments/assets/6ed53bba-919d-4c3e-a3b3-dbcd4473fac1)

```
np.reciprocal(df["Moderate Positive Skew"])
```

![Screenshot 2024-10-07 084647](https://github.com/user-attachments/assets/ead13ca2-0d48-482c-b77e-48394dbe87c9)

```
np.sqrt(df["Highly Positive Skew"])
```

![Screenshot 2024-10-07 084715](https://github.com/user-attachments/assets/15a1d7c9-76d1-438b-ad65-0c137939e9b6)

```
np.square(df["Highly Positive Skew"])
```

![Screenshot 2024-10-07 084741](https://github.com/user-attachments/assets/f4623b9d-36c0-44b0-98a5-4d8a4101f5c6)

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

![Screenshot 2024-10-07 084820](https://github.com/user-attachments/assets/c320aace-5bae-4d44-bc5d-86d846c31596)

```
df.skew()
```

![Screenshot 2024-10-07 084859](https://github.com/user-attachments/assets/4df2f006-b395-4732-8a43-d9aad9fce650)

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```

![Screenshot 2024-10-07 084924](https://github.com/user-attachments/assets/7d5a00e7-1eff-4cd5-9416-7d15dca4c267)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```

![Screenshot 2024-10-07 085000](https://github.com/user-attachments/assets/2cf94502-72a7-4177-a696-0782d7336491)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![Screenshot 2024-10-07 085036](https://github.com/user-attachments/assets/51e158a9-81ad-4bbe-b090-2a4e41be651b)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![Screenshot 2024-10-07 085113](https://github.com/user-attachments/assets/e15d19b4-7848-4e41-8e3d-de63cc39b89a)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![Screenshot 2024-10-07 085152](https://github.com/user-attachments/assets/46655b99-de73-4a8e-8ce3-9800cf746f42)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

![Screenshot 2024-10-07 085223](https://github.com/user-attachments/assets/339dd195-f2ee-4939-90a4-6ef1479ae286)

```
dt=pd.read_csv("titanic_dataset.csv")
dt
```
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```

![Screenshot 2024-10-07 085303](https://github.com/user-attachments/assets/a11bd9ee-7c69-4eac-a27d-6c7ffdcaefe2)

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

![Screenshot 2024-10-07 085327](https://github.com/user-attachments/assets/7f777e97-99a2-4444-99af-b66993ec4c7b)

# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
