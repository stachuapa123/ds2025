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
users = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user', sep="|", index_col=0)
users.head(3)
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
      <th>1</th>
      <td>24</td>
      <td>M</td>
      <td>technician</td>
      <td>85711</td>
    </tr>
    <tr>
      <th>2</th>
      <td>53</td>
      <td>F</td>
      <td>other</td>
      <td>94043</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23</td>
      <td>M</td>
      <td>writer</td>
      <td>32067</td>
    </tr>
  </tbody>
</table>
</div>



### Step 4. See the first 25 entries


```python
users.head(25)
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
      <th>1</th>
      <td>24</td>
      <td>M</td>
      <td>technician</td>
      <td>85711</td>
    </tr>
    <tr>
      <th>2</th>
      <td>53</td>
      <td>F</td>
      <td>other</td>
      <td>94043</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23</td>
      <td>M</td>
      <td>writer</td>
      <td>32067</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24</td>
      <td>M</td>
      <td>technician</td>
      <td>43537</td>
    </tr>
    <tr>
      <th>5</th>
      <td>33</td>
      <td>F</td>
      <td>other</td>
      <td>15213</td>
    </tr>
    <tr>
      <th>6</th>
      <td>42</td>
      <td>M</td>
      <td>executive</td>
      <td>98101</td>
    </tr>
    <tr>
      <th>7</th>
      <td>57</td>
      <td>M</td>
      <td>administrator</td>
      <td>91344</td>
    </tr>
    <tr>
      <th>8</th>
      <td>36</td>
      <td>M</td>
      <td>administrator</td>
      <td>05201</td>
    </tr>
    <tr>
      <th>9</th>
      <td>29</td>
      <td>M</td>
      <td>student</td>
      <td>01002</td>
    </tr>
    <tr>
      <th>10</th>
      <td>53</td>
      <td>M</td>
      <td>lawyer</td>
      <td>90703</td>
    </tr>
    <tr>
      <th>11</th>
      <td>39</td>
      <td>F</td>
      <td>other</td>
      <td>30329</td>
    </tr>
    <tr>
      <th>12</th>
      <td>28</td>
      <td>F</td>
      <td>other</td>
      <td>06405</td>
    </tr>
    <tr>
      <th>13</th>
      <td>47</td>
      <td>M</td>
      <td>educator</td>
      <td>29206</td>
    </tr>
    <tr>
      <th>14</th>
      <td>45</td>
      <td>M</td>
      <td>scientist</td>
      <td>55106</td>
    </tr>
    <tr>
      <th>15</th>
      <td>49</td>
      <td>F</td>
      <td>educator</td>
      <td>97301</td>
    </tr>
    <tr>
      <th>16</th>
      <td>21</td>
      <td>M</td>
      <td>entertainment</td>
      <td>10309</td>
    </tr>
    <tr>
      <th>17</th>
      <td>30</td>
      <td>M</td>
      <td>programmer</td>
      <td>06355</td>
    </tr>
    <tr>
      <th>18</th>
      <td>35</td>
      <td>F</td>
      <td>other</td>
      <td>37212</td>
    </tr>
    <tr>
      <th>19</th>
      <td>40</td>
      <td>M</td>
      <td>librarian</td>
      <td>02138</td>
    </tr>
    <tr>
      <th>20</th>
      <td>42</td>
      <td>F</td>
      <td>homemaker</td>
      <td>95660</td>
    </tr>
    <tr>
      <th>21</th>
      <td>26</td>
      <td>M</td>
      <td>writer</td>
      <td>30068</td>
    </tr>
    <tr>
      <th>22</th>
      <td>25</td>
      <td>M</td>
      <td>writer</td>
      <td>40206</td>
    </tr>
    <tr>
      <th>23</th>
      <td>30</td>
      <td>F</td>
      <td>artist</td>
      <td>48197</td>
    </tr>
    <tr>
      <th>24</th>
      <td>21</td>
      <td>F</td>
      <td>artist</td>
      <td>94533</td>
    </tr>
    <tr>
      <th>25</th>
      <td>39</td>
      <td>M</td>
      <td>engineer</td>
      <td>55107</td>
    </tr>
  </tbody>
</table>
</div>



### Step 5. See the last 10 entries


```python
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
print(len(users))
```

    943
    

#### OR:


```python
users.shape[0]
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
    

#### OR:


```python
users.shape[1]
```




    4



### Step 8. Print the name of all the columns.


```python
for columns in users.columns:
    print(columns)
```

    age
    gender
    occupation
    zip_code
    

#### OR:


```python
users.columns
```




    Index(['age', 'gender', 'occupation', 'zip_code'], dtype='object')



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
users.occupation.head(10)
```




    user_id
    1        technician
    2             other
    3            writer
    4        technician
    5             other
    6         executive
    7     administrator
    8     administrator
    9           student
    10           lawyer
    Name: occupation, dtype: string



#### OR:


```python
users['occupation']
```




    user_id
    1         technician
    2              other
    3             writer
    4         technician
    5              other
               ...      
    939          student
    940    administrator
    941          student
    942        librarian
    943          student
    Name: occupation, Length: 943, dtype: string



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
users.describe(include='all')
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
users['occupation'].describe()
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
# iloc is for integer location, index != iloc (opposites)
t = users['age'].value_counts()
x = t.iloc[t.size-1]
print(t.index[t.size-1], end=' ')
for i in range(t.size-2, 0, -1):
    if t.iloc[i] == x:
        print(t.index[i], end = ' ')
    else:
        break
print('\n',"occured", x, "times")
```

    73 10 66 11 7 
     occured 1 times
    

#### OR:


```python
users['age'].value_counts().tail(5)
```




    age
    7     1
    11    1
    66    1
    10    1
    73    1
    Name: count, dtype: int64



# Exercise 2. - Filtering and Sorting Data

Check out [Euro 12 Exercises Video Tutorial](https://youtu.be/iqk5d48Qisg) to watch a data scientist go through the exercises

This time we are going to pull data directly from the internet.

### Step 1. Import the necessary libraries


```python
import pandas as pd
```

### Step 2. Import the dataset from this [address](https://raw.githubusercontent.com/kflisikowsky/pandas_exercises/refs/heads/main/Euro_2012_stats_TEAM.csv). 

### Step 3. Assign it to a variable called euro12.


```python
euro12 =  pd.read_csv('https://raw.githubusercontent.com/kflisikowsky/pandas_exercises/refs/heads/main/Euro_2012_stats_TEAM.csv')
euro12.head(3)
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
      <th>Team</th>
      <th>Goals</th>
      <th>Shots on target</th>
      <th>Shots off target</th>
      <th>Shooting Accuracy</th>
      <th>% Goals-to-shots</th>
      <th>Total shots (inc. Blocked)</th>
      <th>Hit Woodwork</th>
      <th>Penalty goals</th>
      <th>Penalties not scored</th>
      <th>...</th>
      <th>Saves made</th>
      <th>Saves-to-shots ratio</th>
      <th>Fouls Won</th>
      <th>Fouls Conceded</th>
      <th>Offsides</th>
      <th>Yellow Cards</th>
      <th>Red Cards</th>
      <th>Subs on</th>
      <th>Subs off</th>
      <th>Players Used</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Croatia</td>
      <td>4</td>
      <td>13</td>
      <td>12</td>
      <td>51.9%</td>
      <td>16.0%</td>
      <td>32</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>13</td>
      <td>81.3%</td>
      <td>41</td>
      <td>62</td>
      <td>2</td>
      <td>9</td>
      <td>0</td>
      <td>9</td>
      <td>9</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Czech Republic</td>
      <td>4</td>
      <td>13</td>
      <td>18</td>
      <td>41.9%</td>
      <td>12.9%</td>
      <td>39</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>9</td>
      <td>60.1%</td>
      <td>53</td>
      <td>73</td>
      <td>8</td>
      <td>7</td>
      <td>0</td>
      <td>11</td>
      <td>11</td>
      <td>19</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Denmark</td>
      <td>4</td>
      <td>10</td>
      <td>10</td>
      <td>50.0%</td>
      <td>20.0%</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>10</td>
      <td>66.7%</td>
      <td>25</td>
      <td>38</td>
      <td>8</td>
      <td>4</td>
      <td>0</td>
      <td>7</td>
      <td>7</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 35 columns</p>
</div>



### Step 4. Select only the Goal column.


```python
goal = euro12['Goals']
goal.head(3)
```




    0    4
    1    4
    2    4
    Name: Goals, dtype: int64



### Step 5. How many team participated in the Euro2012?


```python
len(euro12)
```




    16



#### OR:


```python
euro12.shape[0]
```




    16



### Step 6. What is the number of columns in the dataset?


```python
ccol = len(euro12.columns)
print(ccol)
```

    35
    

#### OR:


```python
euro12.shape[1]
```




    35



### Step 7. View only the columns Team, Yellow Cards and Red Cards and assign them to a dataframe called discipline


```python
discipline = euro12[['Team', 'Yellow Cards', 'Red Cards']]
discipline.head(3)
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
      <th>Team</th>
      <th>Yellow Cards</th>
      <th>Red Cards</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Croatia</td>
      <td>9</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Czech Republic</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Denmark</td>
      <td>4</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Step 8. Sort the teams by Red Cards, then to Yellow Cards


```python
sortcards = discipline.sort_values(['Red Cards', 'Yellow Cards'], ascending = False)
sortcards.head(10)
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
      <th>Team</th>
      <th>Yellow Cards</th>
      <th>Red Cards</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>Greece</td>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Poland</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Republic of Ireland</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Italy</td>
      <td>16</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Portugal</td>
      <td>12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Spain</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Croatia</td>
      <td>9</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Czech Republic</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Sweden</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>France</td>
      <td>6</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Step 9. Calculate the mean Yellow Cards given per Team


```python
afterdot = 2
MeanY = euro12['Yellow Cards'].mean()
print(MeanY.round(afterdot))
```

    7.44
    

### Step 10. Filter teams that scored more than 6 goals


```python
goals = euro12[euro12.Goals > 6].sort_values('Goals', ascending = False)
print(goals[['Goals']])
```

        Goals
    13     12
    5      10
    

### Step 11. Select the teams that start with G


```python
Gteams = euro12[euro12.Team.str.startswith('G')]
Gteams[['Team']]
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
      <th>Team</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>Germany</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Greece</td>
    </tr>
  </tbody>
</table>
</div>



### Step 12. Select the first 7 columns


```python
euro12.iloc[:, 0:7].head(3) # or simply :7
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
      <th>Team</th>
      <th>Goals</th>
      <th>Shots on target</th>
      <th>Shots off target</th>
      <th>Shooting Accuracy</th>
      <th>% Goals-to-shots</th>
      <th>Total shots (inc. Blocked)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Croatia</td>
      <td>4</td>
      <td>13</td>
      <td>12</td>
      <td>51.9%</td>
      <td>16.0%</td>
      <td>32</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Czech Republic</td>
      <td>4</td>
      <td>13</td>
      <td>18</td>
      <td>41.9%</td>
      <td>12.9%</td>
      <td>39</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Denmark</td>
      <td>4</td>
      <td>10</td>
      <td>10</td>
      <td>50.0%</td>
      <td>20.0%</td>
      <td>27</td>
    </tr>
  </tbody>
</table>
</div>



### Step 13. Select all columns except the last 3.


```python
euro12.iloc[:, 0:-3].head(3) # or simply :-3
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
      <th>Team</th>
      <th>Goals</th>
      <th>Shots on target</th>
      <th>Shots off target</th>
      <th>Shooting Accuracy</th>
      <th>% Goals-to-shots</th>
      <th>Total shots (inc. Blocked)</th>
      <th>Hit Woodwork</th>
      <th>Penalty goals</th>
      <th>Penalties not scored</th>
      <th>...</th>
      <th>Clean Sheets</th>
      <th>Blocks</th>
      <th>Goals conceded</th>
      <th>Saves made</th>
      <th>Saves-to-shots ratio</th>
      <th>Fouls Won</th>
      <th>Fouls Conceded</th>
      <th>Offsides</th>
      <th>Yellow Cards</th>
      <th>Red Cards</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Croatia</td>
      <td>4</td>
      <td>13</td>
      <td>12</td>
      <td>51.9%</td>
      <td>16.0%</td>
      <td>32</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>10</td>
      <td>3</td>
      <td>13</td>
      <td>81.3%</td>
      <td>41</td>
      <td>62</td>
      <td>2</td>
      <td>9</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Czech Republic</td>
      <td>4</td>
      <td>13</td>
      <td>18</td>
      <td>41.9%</td>
      <td>12.9%</td>
      <td>39</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>10</td>
      <td>6</td>
      <td>9</td>
      <td>60.1%</td>
      <td>53</td>
      <td>73</td>
      <td>8</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Denmark</td>
      <td>4</td>
      <td>10</td>
      <td>10</td>
      <td>50.0%</td>
      <td>20.0%</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>10</td>
      <td>5</td>
      <td>10</td>
      <td>66.7%</td>
      <td>25</td>
      <td>38</td>
      <td>8</td>
      <td>4</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 32 columns</p>
</div>



### Step 14. Present only the Shooting Accuracy from England, Italy and Russia


```python
euro12.loc[euro12['Team'].isin(['England', 'Italy', 'Russia']), ['Team', 'Shooting Accuracy']]
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
      <th>Team</th>
      <th>Shooting Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>England</td>
      <td>50.0%</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Italy</td>
      <td>43.0%</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Russia</td>
      <td>22.5%</td>
    </tr>
  </tbody>
</table>
</div>



#### OR:


```python
euro12.set_index('Team', inplace = True)
euro12.loc[['England', 'Italy', 'Russia'], 'Shooting Accuracy']
```




    Team
    England    50.0%
    Italy      43.0%
    Russia     22.5%
    Name: Shooting Accuracy, dtype: object



# Exercise 3. - GroupBy

### Introduction:

GroupBy can be summarized as Split-Apply-Combine.

Special thanks to: https://github.com/justmarkham for sharing the dataset and materials.

Check out this [Diagram](http://i.imgur.com/yjNkiwL.png)  

Check out [Alcohol Consumption Exercises Video Tutorial](https://youtu.be/az67CMdmS6s) to watch a data scientist go through the exercises


### Step 1. Import the necessary libraries


```python
import pandas as pd
```

### Step 2. Import the dataset from this [address](https://raw.githubusercontent.com/justmarkham/DAT8/master/data/drinks.csv). 

### Step 3. Assign it to a variable called drinks.


```python
drinks = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/drinks.csv', sep=",")
drinks.head(3)
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
      <th>country</th>
      <th>beer_servings</th>
      <th>spirit_servings</th>
      <th>wine_servings</th>
      <th>total_litres_of_pure_alcohol</th>
      <th>continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>AS</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albania</td>
      <td>89</td>
      <td>132</td>
      <td>54</td>
      <td>4.9</td>
      <td>EU</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Algeria</td>
      <td>25</td>
      <td>0</td>
      <td>14</td>
      <td>0.7</td>
      <td>AF</td>
    </tr>
  </tbody>
</table>
</div>



### Step 4. Which continent drinks more beer on average?


```python
drinks.groupby('continent')['beer_servings'].mean().sort_values(ascending=False)
```




    continent
    EU    193.777778
    SA    175.083333
    OC     89.687500
    AF     61.471698
    AS     37.045455
    Name: beer_servings, dtype: float64



### Step 5. For each continent print the statistics for wine consumption.


```python
rnd = 2
drinks.groupby('continent').wine_servings.describe().round(rnd)
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>continent</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AF</th>
      <td>53.0</td>
      <td>16.26</td>
      <td>38.85</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>13.00</td>
      <td>233.0</td>
    </tr>
    <tr>
      <th>AS</th>
      <td>44.0</td>
      <td>9.07</td>
      <td>21.67</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>8.00</td>
      <td>123.0</td>
    </tr>
    <tr>
      <th>EU</th>
      <td>45.0</td>
      <td>142.22</td>
      <td>97.42</td>
      <td>0.0</td>
      <td>59.0</td>
      <td>128.0</td>
      <td>195.00</td>
      <td>370.0</td>
    </tr>
    <tr>
      <th>OC</th>
      <td>16.0</td>
      <td>35.62</td>
      <td>64.56</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>8.5</td>
      <td>23.25</td>
      <td>212.0</td>
    </tr>
    <tr>
      <th>SA</th>
      <td>12.0</td>
      <td>62.42</td>
      <td>88.62</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>12.0</td>
      <td>98.50</td>
      <td>221.0</td>
    </tr>
  </tbody>
</table>
</div>



### Step 6. Print the mean alcohol consumption per continent for every column


```python
beer2 = drinks.groupby('continent').mean(numeric_only=True).round(2)
beer2
# beer2['beer_servings, wine_servings, spirit_servings, total_litres_of_pure_alcohol'.split(', ')].round(2)
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
      <th>beer_servings</th>
      <th>spirit_servings</th>
      <th>wine_servings</th>
      <th>total_litres_of_pure_alcohol</th>
    </tr>
    <tr>
      <th>continent</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AF</th>
      <td>61.47</td>
      <td>16.34</td>
      <td>16.26</td>
      <td>3.01</td>
    </tr>
    <tr>
      <th>AS</th>
      <td>37.05</td>
      <td>60.84</td>
      <td>9.07</td>
      <td>2.17</td>
    </tr>
    <tr>
      <th>EU</th>
      <td>193.78</td>
      <td>132.56</td>
      <td>142.22</td>
      <td>8.62</td>
    </tr>
    <tr>
      <th>OC</th>
      <td>89.69</td>
      <td>58.44</td>
      <td>35.62</td>
      <td>3.38</td>
    </tr>
    <tr>
      <th>SA</th>
      <td>175.08</td>
      <td>114.75</td>
      <td>62.42</td>
      <td>6.31</td>
    </tr>
  </tbody>
</table>
</div>



### Step 7. Print the median alcohol consumption per continent for every column


```python
median1 = drinks.groupby('continent').median(numeric_only=True).round(2)
median1
# median1[['beer_servings', 'spirit_servings', 'wine_servings', 'total_litres_of_pure_alcohol']].round(2)
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
      <th>beer_servings</th>
      <th>spirit_servings</th>
      <th>wine_servings</th>
      <th>total_litres_of_pure_alcohol</th>
    </tr>
    <tr>
      <th>continent</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AF</th>
      <td>32.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>2.30</td>
    </tr>
    <tr>
      <th>AS</th>
      <td>17.5</td>
      <td>16.0</td>
      <td>1.0</td>
      <td>1.20</td>
    </tr>
    <tr>
      <th>EU</th>
      <td>219.0</td>
      <td>122.0</td>
      <td>128.0</td>
      <td>10.00</td>
    </tr>
    <tr>
      <th>OC</th>
      <td>52.5</td>
      <td>37.0</td>
      <td>8.5</td>
      <td>1.75</td>
    </tr>
    <tr>
      <th>SA</th>
      <td>162.5</td>
      <td>108.5</td>
      <td>12.0</td>
      <td>6.85</td>
    </tr>
  </tbody>
</table>
</div>



### Step 8. Print the mean, min and max values for spirit consumption.
#### This time output a DataFrame

#### Per continent:


```python
spirit1 = drinks.groupby('continent').spirit_servings.agg(['mean', 'min', 'max']).round(2)
spirit1
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
      <th>mean</th>
      <th>min</th>
      <th>max</th>
    </tr>
    <tr>
      <th>continent</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AF</th>
      <td>16.34</td>
      <td>0</td>
      <td>152</td>
    </tr>
    <tr>
      <th>AS</th>
      <td>60.84</td>
      <td>0</td>
      <td>326</td>
    </tr>
    <tr>
      <th>EU</th>
      <td>132.56</td>
      <td>0</td>
      <td>373</td>
    </tr>
    <tr>
      <th>OC</th>
      <td>58.44</td>
      <td>0</td>
      <td>254</td>
    </tr>
    <tr>
      <th>SA</th>
      <td>114.75</td>
      <td>25</td>
      <td>302</td>
    </tr>
  </tbody>
</table>
</div>



#### Or in general:


```python
drinks['spirit_servings'].agg(['mean', 'min', 'max'])
```




    mean     80.994819
    min       0.000000
    max     438.000000
    Name: spirit_servings, dtype: float64


