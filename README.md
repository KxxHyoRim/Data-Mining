# Data-Mining


## 기본세팅

```python
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from google.colab import drive


drive.mount('/content/drive')
file_name = '/content/drive/My Drive/Data Mining/seoul_0.csv'

data = pd.read_csv('seoul_0.csv', encoding='cp949')
df = pd.DataFrame(data, columns=['키', '몸무게', '학년', '수축기', '이완기'])
# print(df.shape)
df.head(3)
```

**인코딩**

- `encoding='ISO-8859-1'` 

- 필요 없는것도 있음 (숫자로만 구성된 csv)



## .ipnyb ➡ .py

1. from google.colab import drive 주석처리
2. drive.mount( … ) 코드도 주석처리
3. pd.read_csv() 안의 경로 수정
4.  print()문 추가하기



## 모델 기본

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, 				random_state=0)

clf = clf.fit(X_train, y_train)		# x, y 같이

y_pred = clf.predict(X_test)		# x 만

prob = a.predict_proba(X_test)		# x 만 (각각의 카테고리에 대한 확률 구하기)
									# 1학년일 확률,4학년일 확률 (0.2, 0.8)

clf.score(X_test,y_test)			# x, y 같이

print("Number of mislabled points out of a total %d points : %d " %
      (X_test.shape[0],            # test data size 
       (y_test != y_pred).sum()))  # wrong count


```





## ↔ '행(row)' 제거

**null 이 있는 행 제거**

```python
df_drop_row = df.dropna()				# null이 있는 행 제거( axis 지정 X)
```

```python
df_drop_row.isnull().sum()				# 확인
```

**ex ) 건축년도의 값이 null 인 행 제거**

```python
nan_idx = df[df['건축년도'].isnull()].index
df = df.drop(nan_idx) 
df['건축년도'].unique()
```

**ex ) 건축년도가 0인 값을 가지고 있는 행**

```python
no_year_df = df_drop_row[df_drop_row['건축년도']==0].index
df_drop_row = df_drop_row.drop(no_year_df)		# 행 제거( axis 지정 X)
```



## ↕  '열(column)' 제거

**특정 열 제거**

```python
df1 = df1.drop(columns=['품목별'])
```

**null이 있는 열 제거( axis 지정 )**

```python
df_drop_col = df.dropna(axis = 1)		
```



### 데이터 Summery / 분포 / 구성

```python
''' 데이터 분포 확인'''
grade = data['학년']
grade.value_counts()
```

```python
''' 구성 확인 '''
df.describe()		# 전체
df[‘특정행’].uniqe()
```

```python
# data type 확인 (데이터 타입 확인)
df.dtypes
```



## 🕐훈련 속도 비교

```python
import time

start_time = time.time()
clf = MLPClassifier().fit
end_time = time.time()
print("훈련 속도 : ", end_time - start_time )
```



## ↗소수점 반올림

```python
round(n,3)
```



