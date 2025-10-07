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
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
<img width="515" height="441" alt="Screenshot 2025-10-07 155852" src="https://github.com/user-attachments/assets/4d8fd55e-3b3a-4b1a-a97c-80f4f0d60c33" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
<img width="365" height="234" alt="Screenshot 2025-10-07 160105" src="https://github.com/user-attachments/assets/3c2a46cc-7bec-40db-8cb5-c8941436703e" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
<img width="596" height="454" alt="Screenshot 2025-10-07 160221" src="https://github.com/user-attachments/assets/cac678d1-65ea-4c27-82a7-0e3ba28d19d3" />

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
<img width="499" height="451" alt="Screenshot 2025-10-07 160335" src="https://github.com/user-attachments/assets/1d02580b-2b62-4857-a4c4-6acdf4ed0827" />

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
<img width="619" height="444" alt="Screenshot 2025-10-07 160501" src="https://github.com/user-attachments/assets/f1a4d8e7-7f20-4ff7-b696-21b466d4c47a" />

```
pd.get_dummies(df2,columns=["nom_0"])
```
<img width="875" height="445" alt="Screenshot 2025-10-07 160544" src="https://github.com/user-attachments/assets/91e6b29c-301c-41d9-bba8-affd0ac3294b" />

```

from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
<img width="668" height="438" alt="Screenshot 2025-10-07 160634" src="https://github.com/user-attachments/assets/0cff66d6-8c08-4ab5-8594-f63873741a35" />

```

be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```

<img width="707" height="450" alt="Screenshot 2025-10-07 160735" src="https://github.com/user-attachments/assets/7e8a56d1-3326-47bf-ac3a-b72955ce055d" />

```

dfb=pd.concat([df,nd],axis=1)
dfb
```
<img width="959" height="454" alt="Screenshot 2025-10-07 160819" src="https://github.com/user-attachments/assets/d00b9b8b-2043-4a08-b4f0-ee04c06a8d30" />

```

from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
<img width="792" height="450" alt="Screenshot 2025-10-07 160906" src="https://github.com/user-attachments/assets/a1709c7f-78ae-49b7-9ea3-a56d453928e4" />

```

import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
<img width="1138" height="528" alt="Screenshot 2025-10-07 160950" src="https://github.com/user-attachments/assets/d727c581-3b3e-48f4-b3f7-b1dc56570fe0" />

```

df.skew()
```
<img width="503" height="254" alt="Screenshot 2025-10-07 161032" src="https://github.com/user-attachments/assets/b9d87428-9f1b-41e8-a477-f2473c84cfaf" />

```

np.log(df["Highly Positive Skew"])
```
<img width="457" height="573" alt="Screenshot 2025-10-07 161123" src="https://github.com/user-attachments/assets/306bb462-c961-41ad-a4ca-71e0e11d347e" />

```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="475" height="565" alt="Screenshot 2025-10-07 161205" src="https://github.com/user-attachments/assets/79ba27a3-5c05-4923-b3ca-75f93bf4972d" />

```
np.sqrt(df["Highly Positive Skew"])
```
<img width="465" height="570" alt="Screenshot 2025-10-07 161253" src="https://github.com/user-attachments/assets/3db5d90f-ef18-4641-8400-f26f545ca3ae" />

```

np.square(df["Highly Positive Skew"])
```
<img width="449" height="570" alt="Screenshot 2025-10-07 161349" src="https://github.com/user-attachments/assets/6e16fc9d-0d99-451a-b464-2662973ab8e7" />

```

df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="1350" height="531" alt="Screenshot 2025-10-07 161434" src="https://github.com/user-attachments/assets/ba6d92df-a7eb-4a89-aa09-479100b86a5e" />

```
df.skew()
```
<img width="511" height="304" alt="Screenshot 2025-10-07 161516" src="https://github.com/user-attachments/assets/36fe79e2-e642-4a8b-97ec-9dabf3a5e9a7" />

```

df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
<img width="679" height="345" alt="Screenshot 2025-10-07 161554" src="https://github.com/user-attachments/assets/21f7e307-ba01-45de-803b-780458b89b53" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
<img width="1768" height="523" alt="Screenshot 2025-10-07 161658" src="https://github.com/user-attachments/assets/0d45b58a-fed8-45aa-bcc3-ec9d72109ac7" />

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="861" height="553" alt="Screenshot 2025-10-07 161748" src="https://github.com/user-attachments/assets/61ce84e5-ee17-4d3a-94c9-807176704244" />


```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
<img width="864" height="563" alt="Screenshot 2025-10-07 161838" src="https://github.com/user-attachments/assets/4e708804-ed66-44f4-973d-9e333fafb68b" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="855" height="557" alt="Screenshot 2025-10-07 161926" src="https://github.com/user-attachments/assets/2b8d2cef-bad9-4529-ba33-f85ff3cd1250" />

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
<img width="888" height="561" alt="Screenshot 2025-10-07 162009" src="https://github.com/user-attachments/assets/30bf1613-3b37-4705-a467-7a734503e117" />

```

dt=pd.read_csv("/content/titanic_dataset.csv")
dt
```
<img width="1571" height="532" alt="Screenshot 2025-10-07 162054" src="https://github.com/user-attachments/assets/88cea1f5-bee4-4b22-8c3d-e1bee41aada9" />

```

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()
```
<img width="900" height="559" alt="Screenshot 2025-10-07 162138" src="https://github.com/user-attachments/assets/91f957e9-e0fd-4524-a88a-0a44ee8b6c83" />

```

sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
<img width="946" height="564" alt="Screenshot 2025-10-07 162216" src="https://github.com/user-attachments/assets/6975ec99-2332-4bf1-8380-e30cf2af6683" />


# RESULT:
       THE GIVEN DATA EXECUTED SUCCESSFULLY

       
