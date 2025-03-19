# Exercise 1. - Getting and Knowing your Data

This time we are going to pull data directly from the internet.
Special thanks to: https://github.com/justmarkham for sharing the dataset and materials.

Check out [Occupation Exercises Video Tutorial](https://www.youtube.com/watch?v=W8AB5s-L3Rw&list=PLgJhDSE2ZLxaY_DigHeiIDC1cD09rXgJv&index=4) to watch a data scientist go through the exercises

### Step 1. Import the necessary libraries


```python
import pandas as pd
```

### Step 2. Import the dataset from this [address](https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user). 

### Step 3. Assign it to a variable called users and use the 'user_id' as index


```python
import pandas as pd
users = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user', sep="|", index_col=0)
users.head(3)
```

### Step 4. See the first 25 entries


```python
import pandas as pd
users = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user', sep="|", index_col=0)
users.head(25)
```

### Step 5. See the last 10 entries


```python
import pandas as pd
users = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user', sep="|", index_col=0)
users.tail(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>gender</th>
      <th>occupation</th>
      <th>zip_code</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>934</th>
      <td>61</td>
      <td>M</td>
      <td>engineer</td>
      <td>22902</td>
    </tr>
    <tr>
      <th>935</th>
      <td>42</td>
      <td>M</td>
      <td>doctor</td>
      <td>66221</td>
    </tr>
    <tr>
      <th>936</th>
      <td>24</td>
      <td>M</td>
      <td>other</td>
      <td>32789</td>
    </tr>
    <tr>
      <th>937</th>
      <td>48</td>
      <td>M</td>
      <td>educator</td>
      <td>98072</td>
    </tr>
    <tr>
      <th>938</th>
      <td>38</td>
      <td>F</td>
      <td>technician</td>
      <td>55038</td>
    </tr>
    <tr>
      <th>939</th>
      <td>26</td>
      <td>F</td>
      <td>student</td>
      <td>33319</td>
    </tr>
    <tr>
      <th>940</th>
      <td>32</td>
      <td>M</td>
      <td>administrator</td>
      <td>02215</td>
    </tr>
    <tr>
      <th>941</th>
      <td>20</td>
      <td>M</td>
      <td>student</td>
      <td>97229</td>
    </tr>
    <tr>
      <th>942</th>
      <td>48</td>
      <td>F</td>
      <td>librarian</td>
      <td>78209</td>
    </tr>
    <tr>
      <th>943</th>
      <td>22</td>
      <td>M</td>
      <td>student</td>
      <td>77841</td>
    </tr>
  </tbody>
</table>
</div>



### Step 6. What is the number of observations in the dataset?


```python
import pandas as pd
users = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user', sep="|", index_col=0)
print(len(users))
```

    943
    

### Step 7. What is the number of columns in the dataset?


```python
i=0
for columns in users.columns:
    i+=1
print(i)
```

    4
    

### Step 8. Print the name of all the columns.


```python
for columns in users.columns:
    print(columns)
```

### Step 9. How is the dataset indexed?


```python
users.index
```




    Index([  1,   2,   3,   4,   5,   6,   7,   8,   9,  10,
           ...
           934, 935, 936, 937, 938, 939, 940, 941, 942, 943],
          dtype='int64', name='user_id', length=943)



### Step 10. What is the data type of each column?


```python
users.dtypes
users['gender'] = users['gender'].convert_dtypes(str)
users['occupation'] = users['occupation'].convert_dtypes(str)
users['zip_code'] = users['zip_code'].convert_dtypes(str)
users.dtypes

```




    age                    int64
    gender        string[python]
    occupation    string[python]
    zip_code      string[python]
    dtype: object



### Step 11. Print only the occupation column


```python
#df = pd.DataFrame(users)
#print(df['occupation'].head(3))
users.occupation.head(10)
```

    user_id
    1    technician
    2         other
    3        writer
    Name: occupation, dtype: object
    

### Step 12. How many different occupations are in this dataset?


```python
users.occupation.nunique()
```




    21



### Step 13. What is the most frequent occupation?


```python
users.occupation.value_counts().head(1)
```




    occupation
    student    196
    Name: count, dtype: Int64



### Step 14. Summarize the DataFrame.


```python
users.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>943.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>34.051962</td>
    </tr>
    <tr>
      <th>std</th>
      <td>12.192740</td>
    </tr>
    <tr>
      <th>min</th>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>43.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>73.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Step 15. Summarize all the columns


```python
sall = users.describe(include='all')
sall
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>gender</th>
      <th>occupation</th>
      <th>zip_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>943.000000</td>
      <td>943</td>
      <td>943</td>
      <td>943</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>2</td>
      <td>21</td>
      <td>795</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>M</td>
      <td>student</td>
      <td>55414</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>670</td>
      <td>196</td>
      <td>9</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>34.051962</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>12.192740</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>7.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>25.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>31.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>43.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>73.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### Step 16. Summarize only the occupation column


```python
occs = users['occupation'].describe()
occs
```




    count         943
    unique         21
    top       student
    freq          196
    Name: occupation, dtype: object



### Step 17. What is the mean age of users?


```python
afterdot = 2 # number of digits after the decimal point
MeanAge = users['age'].mean()
print(MeanAge.round(afterdot))
```

    34.05
    

### Step 18. What is the age with least occurrence?


```python
i = 1
t = []
b = 1
least = 0
while(b):
    for users['age'] in users['age']:
        if users['age'] == i:
            b=0
            least = i
            break 
for users['age'] in users['age']:
    if users['age'] == least:
        t.append(users['age'])
print(t)
print(i)
#users['age'].value_counts().tail(10)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_22808\3240434711.py in ?()
          3 b = 1
          4 least = 0
          5 while(b):
          6     for users['age'] in users['age']:
    ----> 7         if users['age'] == i:
          8             b=0
          9             least = i
         10             break
    

    c:\Users\mmapa\Desktop\numf\en1\Lib\site-packages\pandas\core\generic.py in ?(self)
       1575     @final
       1576     def __nonzero__(self) -> NoReturn:
    -> 1577         raise ValueError(
       1578             f"The truth value of a {type(self).__name__} is ambiguous. "
       1579             "Use a.empty, a.bool(), a.item(), a.any() or a.all()."
       1580         )
    

    ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().

