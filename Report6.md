---
title: Bivariate Statistics
subtitle: Foundations of Statistical Analysis in Python
abstract: This notebook explores bivariate relationships through linear correlations, highlighting their strengths and limitations. Practical examples and visualizations are provided to help users understand and apply these statistical concepts effectively.
author:
  - name: Karol Flisikowski
    affiliations: 
      - Gdansk University of Technology
      - Chongqing Technology and Business University
    orcid: 0000-0002-4160-1297
    email: karol@ctbu.edu.cn
date: 2025-05-03
---

## Goals of this lecture

There are many ways to *describe* a distribution. 

Here we will discuss:
- Measurement of the relationship between distributions using **linear, rank correlations**.
- Measurement of the relationship between qualitative variables using **contingency**.

## Importing relevant libraries


```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns ### importing seaborn
import pandas as pd
import scipy.stats as ss
```


```python
%matplotlib inline 
%config InlineBackend.figure_format = 'retina'
```


```python
import pandas as pd
df_pokemon = pd.read_csv("pokemon.csv")
```

## Describing *bivariate* data with correlations

- So far, we've been focusing on *univariate data*: a single distribution.
- What if we want to describe how *two distributions* relate to each other?
   - For today, we'll focus on *continuous distributions*.

### Bivariate relationships: `height`

- A classic example of **continuous bivariate data** is the `height` of a `parent` and `child`.  
- [These data were famously collected by Karl Pearson](https://www.kaggle.com/datasets/abhilash04/fathersandsonheight).


```python
df_height = pd.read_csv("height.csv")
df_height.head(2)
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
      <th>Father</th>
      <th>Son</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>65.0</td>
      <td>59.8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>63.3</td>
      <td>63.2</td>
    </tr>
  </tbody>
</table>
</div>



#### Plotting Pearson's height data


```python
sns.scatterplot(data = df_height, x = "Father", y = "Son", alpha = .5);
```


    
![png](Report6_files/Report6_10_0.png)
    


### Introducing linear correlations

> A **correlation coefficient** is a number between $[–1, 1]$ that describes the relationship between a pair of variables.

Specifically, **Pearson's correlation coefficient** (or Pearson's $r$) describes a (presumed) *linear* relationship.

Two key properties:

- **Sign**: whether a relationship is positive (+) or negative (–).  
- **Magnitude**: the strength of the linear relationship.

$$
r = \frac{ \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y}) }{ \sqrt{ \sum_{i=1}^{n} (x_i - \bar{x})^2 } \sqrt{ \sum_{i=1}^{n} (y_i - \bar{y})^2 } }
$$

Where:
- $r$ - Pearson correlation coefficient
- $x_i$, $y_i$ - values of the variables
- $\bar{x}$, $\bar{y}$ - arithmetic means
- $n$ - number of observations

Pearson's correlation coefficient measures the strength and direction of the linear relationship between two continuous variables. Its value ranges from -1 to 1:
- 1 → perfect positive linear correlation
- 0 → no linear correlation
- -1 → perfect negative linear correlation

This coefficient does not tell about nonlinear correlations and is sensitive to outliers.

### Calculating Pearson's $r$ with `scipy`

`scipy.stats` has a function called `pearsonr`, which will calculate this relationship for you.

Returns two numbers:

- $r$: the correlation coefficent.  
- $p$: the **p-value** of this correlation coefficient, i.e., whether it's *significantly different* from `0`.


```python
ss.pearsonr(df_height['Father'], df_height['Son'])
```




    PearsonRResult(statistic=np.float64(0.5011626808075912), pvalue=np.float64(1.272927574366214e-69))



#### Check-in

Using `scipy.stats.pearsonr` (here, `ss.pearsonr`), calculate Pearson's $r$ for the relationship between the `Attack` and `Defense` of Pokemon.

- Is this relationship positive or negative?  
- How strong is this relationship?


```python
rad = ss.pearsonr(df_pokemon['Attack'], df_pokemon['Defense'])
float(rad[0].round(2))
```




    0.44



#### Solution


```python
ss.pearsonr(df_pokemon['Attack'], df_pokemon['Defense'])
```




    PearsonRResult(statistic=np.float64(0.4386870551184896), pvalue=np.float64(5.858479864289521e-39))



#### Check-in

Pearson'r $r$ measures the *linear correlation* between two variables. Can anyone think of potential limitations to this approach?

### Limitations of Pearson's $r$

- Pearson's $r$ *presumes* a linear relationship and tries to quantify its strength and direction.  
- But many relationships are **non-linear**!  
- Unless we visualize our data, relying only on Pearson'r $r$ could mislead us.

#### Non-linear data where $r = 0$


```python
x = np.arange(1, 40)
y = np.sin(x)
p = sns.lineplot(x = x, y = y)
```


    
![png](Report6_files/Report6_23_0.png)
    



```python
### r is close to 0, despite there being a clear relationship!
ss.pearsonr(x, y)
```




    PearsonRResult(statistic=np.float64(-0.04067793461845843), pvalue=np.float64(0.8057827185936625))



#### When $r$ is invariant to the real relationship

All these datasets have roughly the same **correlation coefficient**.


```python
df_anscombe = sns.load_dataset("anscombe")
sns.relplot(data = df_anscombe, x = "x", y = "y", col = "dataset");
```


    
![png](Report6_files/Report6_26_0.png)
    



```python
# Compute correlation matrix
corr = df_pokemon.corr(numeric_only=True)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a heatmap
sns.heatmap(corr, 
            annot=True,         # Show correlation coefficients
            fmt=".2f",          # Format for coefficients
            cmap="coolwarm",    # Color palette
            vmin=-1, vmax=1,    # Fixed scale
            square=True,        # Make cells square
            linewidths=0.5,     # Line width between cells
            cbar_kws={"shrink": .75})  # Colorbar shrink

# Title and layout
plt.title("Correlation Heatmap", fontsize=16)
plt.tight_layout()

# Show plot
plt.show()
```


    
![png](Report6_files/Report6_27_0.png)
    


## Rank Correlations

Rank correlations are measures of the strength and direction of a monotonic (increasing or decreasing) relationship between two variables. Instead of numerical values, they use ranks, i.e., positions in an ordered set.

They are less sensitive to outliers and do not require linearity (unlike Pearson's correlation).

### Types of Rank Correlations

1. $ρ$ (rho) **Spearman's**
- Based on the ranks of the data.
- Value: from –1 to 1.
- Works well for monotonic but non-linear relationships.

$$
\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}
$$

Where:
- $d_i$ – differences between the ranks of observations,
- $n$ – number of observations.

2. $τ$ (tau) **Kendall's**
- Measures the number of concordant vs. discordant pairs.
- More conservative than Spearman's – often yields smaller values.
- Also ranges from –1 to 1.

$$
\tau = \frac{(C - D)}{\frac{1}{2}n(n - 1)}
$$

Where:
- $τ$ — Kendall's correlation coefficient,
- $C$ — number of concordant pairs,
- $D$ — number of discordant pairs,
- $n$ — number of observations,
- $\frac{1}{2}n(n - 1)$ — total number of possible pairs of observations.

What are concordant and discordant pairs?
- Concordant pair: if $x_i$ < $x_j$ and $y_i$ < $y_j$, or $x_i$ > $x_j$ and $y_i$ > $y_j$.
- Discordant pair: if $x_i$ < $x_j$ and $y_i$ > $y_j$, or $x_i$ > $x_j$ and $y_i$ < $y_j$.

### When to use rank correlations?
- When the data are not normally distributed.
- When you suspect a non-linear but monotonic relationship.
- When you have rank correlations, such as grades, ranking, preference level.

| Correlation type | Description | When to use |
|------------------|-----------------------------------------------------|----------------------------------------|
| Spearman's (ρ) | Monotonic correlation, based on ranks | When data are nonlinear or have outliers |
| Kendall's (τ) | Counts the proportion of congruent and incongruent pairs | When robustness to ties is important |

### Interpretation of correlation values

| Range of values | Correlation interpretation |
|------------------|----------------------------------|
| 0.8 - 1.0 | very strong positive |
| 0.6 - 0.8 | strong positive |
| 0.4 - 0.6 | moderate positive |
| 0.2 - 0.4 | weak positive |
| 0.0 - 0.2 | very weak or no correlation |
| < 0 | similarly - negative correlation |


```python
# Compute Kendall rank correlation
corr_kendall = df_pokemon.corr(method='kendall', numeric_only=True)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a heatmap
sns.heatmap(corr, 
            annot=True,         # Show correlation coefficients
            fmt=".2f",          # Format for coefficients
            cmap="coolwarm",    # Color palette
            vmin=-1, vmax=1,    # Fixed scale
            square=True,        # Make cells square
            linewidths=0.5,     # Line width between cells
            cbar_kws={"shrink": .75})  # Colorbar shrink

# Title and layout
plt.title("Correlation Heatmap", fontsize=16)
plt.tight_layout()

# Show plot
plt.show()
```


    
![png](Report6_files/Report6_32_0.png)
    


### Comparison of Correlation Coefficients

| Property                | Pearson (r)                   | Spearman (ρ)                        | Kendall (τ)                          |
|-------------------------|-------------------------------|--------------------------------------|---------------------------------------|
| What it measures?       | Linear relationship           | Monotonic relationship (based on ranks) | Monotonic relationship (based on pairs) |
| Data type               | Quantitative, normal distribution | Ranks or ordinal/quantitative data  | Ranks or ordinal/quantitative data   |
| Sensitivity to outliers | High                          | Lower                               | Low                                   |
| Value range             | –1 to 1                       | –1 to 1                             | –1 to 1                               |
| Requires linearity      | Yes                           | No                                  | No                                    |
| Robustness to ties      | Low                           | Medium                              | High                                  |
| Interpretation          | Strength and direction of linear relationship | Strength and direction of monotonic relationship | Proportion of concordant vs discordant pairs |
| Significance test       | Yes (`scipy.stats.pearsonr`)  | Yes (`spearmanr`)                   | Yes (`kendalltau`)                   |

Brief summary:
- Pearson - best when the data are normal and the relationship is linear.
- Spearman - works better for non-linear monotonic relationships.
- Kendall - more conservative, often used in social research, less sensitive to small changes in data.

### Your Turn

For the Pokemon dataset, find the pairs of variables that are most appropriate for using one of the quantitative correlation measures. Calculate them, then visualize them.


```python
from scipy.stats import pearsonr, spearmanr, kendalltau

pear, pearson_p = pearsonr(df_pokemon['Attack'], df_pokemon['Total'])
spear, spearman_p = spearmanr(df_pokemon['Attack'], df_pokemon['Total'])
kend, kendall_p = kendalltau(df_pokemon['Attack'], df_pokemon['Total'])

print("Pearson: ", pear)

print("Spearman: ", spear)

print("Kendall: ", kend)

sns.scatterplot(x=df_pokemon['Attack'], y=df_pokemon['Total'])

sns.regplot(x=df_pokemon['Attack'], y=df_pokemon['Total'], scatter=False, color='red')



plt.legend()
plt.title("Attack vs Total with Correlation Trend Lines")
plt.show()
```

    Pearson:  0.7362106519949294
    Spearman:  0.7201309606515207
    Kendall:  0.5398292618765553
    


    
![png](Report6_files/Report6_36_1.png)
    


## Correlation of Qualitative Variables

A categorical variable is one that takes descriptive values ​​that represent categories—e.g. Pokémon type (Fire, Water, Grass), gender, status (Legendary vs. Normal), etc.

Such variables cannot be analyzed directly using correlation methods for numbers (Pearson, Spearman, Kendall). Other techniques are used instead.

### Contingency Table

A contingency table is a special cross-tabulation table that shows the frequency (i.e., the number of cases) for all possible combinations of two categorical variables.

It is a fundamental tool for analyzing relationships between qualitative features.

#### Chi-Square Test of Independence

The Chi-Square test checks whether there is a statistically significant relationship between two categorical variables.

Concept:

We compare:
- observed values (from the contingency table),
- with expected values, assuming the variables are independent.

$$
\chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
$$

Where:
- $O_{ij}$ – observed count in cell ($i$, $j$),
- $E_{ij}$ – expected count in cell ($i$, $j$), assuming independence.

### Example: Calculating Expected Values and Chi-Square Statistic in Python

Here’s how you can calculate the **expected values** and **Chi-Square statistic (χ²)** step by step using Python.

---

#### Step 1: Create the Observed Contingency Table
We will use the Pokémon example:

| Type 1 | Legendary = False | Legendary = True | Total |
|--------|-------------------|------------------|-------|
| Fire   | 18                | 5                | 23    |
| Water  | 25                | 3                | 28    |
| Grass  | 20                | 2                | 22    |
| Total  | 63                | 10               | 73    |


```python
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# Observed values (contingency table)
observed = np.array([
    [18, 5],  # Fire
    [25, 3],  # Water
    [20, 2]   # Grass
])

# Convert to DataFrame for better visualization
observed_df = pd.DataFrame(
    observed,
    columns=["Legendary = False", "Legendary = True"],
    index=["Fire", "Water", "Grass"]
)
print("Observed Table:")
print(observed_df)
```

    Observed Table:
           Legendary = False  Legendary = True
    Fire                  18                 5
    Water                 25                 3
    Grass                 20                 2
    

Step 2: Calculate Expected Values
The expected values are calculated using the formula:

$$ E_{ij} = \frac{\text{Row Total} \times \text{Column Total}}{\text{Grand Total}} $$

You can calculate this manually or use scipy.stats.chi2_contingency, which automatically computes the expected values.


```python
# Perform Chi-Square test
chi2, p, dof, expected = chi2_contingency(observed)

# Convert expected values to DataFrame for better visualization
expected_df = pd.DataFrame(
    expected,
    columns=["Legendary = False", "Legendary = True"],
    index=["Fire", "Water", "Grass"]
)
print("\nExpected Table:")
print(expected_df)
```

    
    Expected Table:
           Legendary = False  Legendary = True
    Fire           19.849315          3.150685
    Water          24.164384          3.835616
    Grass          18.986301          3.013699
    

Step 3: Calculate the Chi-Square Statistic
The Chi-Square statistic is calculated using the formula:

$$ \chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}} $$

This is done automatically by scipy.stats.chi2_contingency, but you can also calculate it manually:


```python
# Manual calculation of Chi-Square statistic
chi2_manual = np.sum((observed - expected) ** 2 / expected)
print(f"\nChi-Square Statistic (manual): {chi2_manual:.4f}")
```

    
    Chi-Square Statistic (manual): 1.8638
    

Step 4: Interpret the Results
The chi2_contingency function also returns:

p-value: The probability of observing the data if the null hypothesis (independence) is true.
Degrees of Freedom (dof): Calculated as (rows - 1) * (columns - 1).


```python
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"p-value: {p:.4f}")
print(f"Degrees of Freedom: {dof}")
```

    
    Chi-Square Statistic: 1.8638
    p-value: 0.3938
    Degrees of Freedom: 2
    

**Interpretation of the Chi-Square Test Result:**

| Value               | Meaning                                         |
|---------------------|-------------------------------------------------|
| High χ² value       | Large difference between observed and expected values |
| Low p-value         | Strong basis to reject the null hypothesis of independence |
| p < 0.05            | Statistically significant relationship between variables |

### Qualitative Correlations

#### Cramér's V

**Cramér's V** is a measure of the strength of association between two categorical variables. It is based on the Chi-Square test but scaled to a range of 0–1, making it easier to interpret the strength of the relationship.

$$
V = \sqrt{ \frac{\chi^2}{n \cdot (k - 1)} }
$$

Where:
- $\chi^2$ – Chi-Square test statistic,
- $n$ – number of observations,
- $k$ – the smaller number of categories (rows/columns) in the contingency table.

---

#### Phi Coefficient ($φ$)

Application:
- Both variables must be dichotomous (e.g., Yes/No, 0/1), meaning the table must have the smallest size of **2×2**.
- Ideal for analyzing relationships like gender vs purchase, type vs legendary.

$$
\phi = \sqrt{ \frac{\chi^2}{n} }
$$

Where:
- $\chi^2$ – Chi-Square test statistic for a 2×2 table,
- $n$ – number of observations.

---

#### Tschuprow’s T

**Tschuprow’s T** is a measure of association similar to **Cramér's V**, but it has a different scale. It is mainly used when the number of categories in the two variables differs. This is a more advanced measure applicable to a broader range of contingency tables.

$$
T = \sqrt{\frac{\chi^2}{n \cdot (k - 1)}}
$$

Where:
- $\chi^2$ – Chi-Square test statistic,
- $n$ – number of observations,
- $k$ – the smaller number of categories (rows or columns) in the contingency table.

Application: Tschuprow’s T is useful when dealing with contingency tables with varying numbers of categories in rows and columns.

---

### Summary - Qualitative Correlations

| Measure            | What it measures                                       | Application                     | Value Range     | Strength Interpretation       |
|--------------------|--------------------------------------------------------|---------------------------------|------------------|-------------------------------|
| **Cramér's V**     | Strength of association between nominal variables      | Any categories                  | 0 – 1           | 0.1–weak, 0.3–moderate, >0.5–strong |
| **Phi ($φ$)**      | Strength of association in a **2×2** table             | Two binary variables            | -1 – 1          | Similar to correlation        |
| **Tschuprow’s T**  | Strength of association, alternative to Cramér's V     | Tables with similar category counts | 0 – 1      | Less commonly used            |
| **Chi² ($χ²$)**    | Statistical test of independence                       | All categorical variables       | 0 – ∞           | Higher values indicate stronger differences |

### Example

Let's investigate whether the Pokémon's type (type_1) is affected by whether the Pokémon is legendary.

We'll use the **scipy** library.

This library already has built-in functions for calculating various qualitative correlation measures.


```python
from scipy.stats.contingency import association

# Contingency table:
ct = pd.crosstab(df_pokemon["Type 1"], df_pokemon["Legendary"])

# Calculating Cramér's V measure
V = association(ct, method="cramer") # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.association.html#association

print(f"Cramer's V: {V}") # interpret!
```

    Cramer's V: 0.3361928228447545
    

### Your turn

What visualization would be most appropriate for presenting a quantitative, ranked, and qualitative relationship?

Try to think about which pairs of variables could have which type of analysis based on the Pokemon data.

---


```python
df_pokemon
# quantitative - scatterplots
# Attack and Defense
sns.scatterplot(data = df_pokemon, x = "Attack", y = "Defense", alpha = .5)
sns.regplot(x=df_pokemon['Attack'], y=df_pokemon['Defense'], scatter=False, color='red')
c1 = ss.pearsonr(df_pokemon['Attack'], df_pokemon['Defense'])
print("correlation between Attack and Defense", float(c1[0]))
```

    correlation between Attack and Defense 0.4386870551184896
    


    
![png](Report6_files/Report6_52_1.png)
    


There is no Ordinal scale variable in df_pokemon so we cannot make a ranked type relationship



```python
df_pokemon
# Type 1(Grass, rock, Psychic) to Generation (1,2,3)
Qual = df_pokemon[df_pokemon['Generation']<=3]
Qual = Qual[Qual['Type 1'].isin(['Grass', 'Rock', 'Psychic'])]
print("total rows to analyse: ", Qual.shape[0])
Qt = pd.crosstab(Qual['Type 1'], Qual['Generation'])
Qtmar = pd.crosstab(Qual['Type 1'], Qual['Generation'], margins=True)
#print(table)
import tabulate
print(tabulate.tabulate(Qtmar, headers='keys', tablefmt='github'))
```

    total rows to analyse:  88
    | Type 1   |   1 |   2 |   3 |   All |
    |----------|-----|-----|-----|-------|
    | Grass    |  13 |   9 |  13 |    35 |
    | Psychic  |  11 |   7 |  12 |    30 |
    | Rock     |  10 |   5 |   8 |    23 |
    | All      |  34 |  21 |  33 |    88 |
    


```python
chi2q, pq, dofq, expectedq = ss.chi2_contingency(Qtmar)

# QtEx = pd.DataFrame(expected, 
#                            index=Qt.index, 
#                            columns=Qt.columns)
# print("Expected Table:")
# print(tabulate.tabulate(QtEx, headers='keys', tablefmt='github'))
# chi2Qm = np.sum(np.sum((Qt - QtEx) ** 2 / QtEx))
# print("Chi statistic for this data (manual): ", chi2Qm)
print("Chi statistic for this data: ", chi2q)
V2 = association(Qt, method="cramer")

print(f"Cramer's V for this data: {V2}")
```

    Chi statistic for this data:  0.38364099262893575
    Cramer's V for this data: 0.046688096632973394
    


```python
# Qualitative
from statsmodels.graphics import mosaicplot

_, ax = plt.subplots(figsize=(16, 8))
mosaicplot.mosaic(df_pokemon, ['Type 1', 'Legendary'], ax=ax)
plt.title('Type 1 vs Legendary', pad=20)
plt.show()
```


    
![png](Report6_files/Report6_56_0.png)
    


## Heatmaps for qualitative correlations


```python
# git clone https://github.com/ayanatherate/dfcorrs.git
# cd dfcorrs 
# pip install -r requirements.txt

from dfcorrs.cramersvcorr import Cramers

cram=Cramers()
# cram.corr(df_pokemon)
cram.corr(df_pokemon, plot_htmp=True)
```



## Your turn!

Load the "sales" dataset and perform the bivariate analysis together with necessary plots. Remember about to run data preprocessing before the analysis.


```python
df_sales = pd.read_excel("sales.xlsx")
df_sales
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
      <th>Date</th>
      <th>Store_Type</th>
      <th>City_Type</th>
      <th>Day_Temp</th>
      <th>No_of_Customers</th>
      <th>Sales</th>
      <th>Product_Quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-10-01</td>
      <td>1</td>
      <td>1</td>
      <td>30.0</td>
      <td>100.0</td>
      <td>3112.0</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-10-02</td>
      <td>2</td>
      <td>1</td>
      <td>32.0</td>
      <td>115.0</td>
      <td>3682.0</td>
      <td>A</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-10-03</td>
      <td>3</td>
      <td>3</td>
      <td>31.0</td>
      <td>NaN</td>
      <td>2774.0</td>
      <td>A</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-10-04</td>
      <td>1</td>
      <td>2</td>
      <td>29.0</td>
      <td>105.0</td>
      <td>3182.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-10-05</td>
      <td>1</td>
      <td>2</td>
      <td>33.0</td>
      <td>104.0</td>
      <td>1368.0</td>
      <td>B</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2020-10-07</td>
      <td>2</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>B</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2020-11-24</td>
      <td>2</td>
      <td>3</td>
      <td>26.0</td>
      <td>90.0</td>
      <td>4232.0</td>
      <td>C</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2020-11-25</td>
      <td>3</td>
      <td>3</td>
      <td>NaN</td>
      <td>96.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2020-11-26</td>
      <td>2</td>
      <td>2</td>
      <td>27.0</td>
      <td>100.0</td>
      <td>2356.0</td>
      <td>B</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2020-11-28</td>
      <td>3</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>A</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2020-11-29</td>
      <td>1</td>
      <td>1</td>
      <td>23.0</td>
      <td>94.0</td>
      <td>1254.0</td>
      <td>A</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2020-11-30</td>
      <td>1</td>
      <td>1</td>
      <td>22.0</td>
      <td>91.0</td>
      <td>4232.0</td>
      <td>A</td>
    </tr>
  </tbody>
</table>
</div>




```python
import missingno as msno

msno.matrix(df_sales)
```




    <Axes: >




    
![png](Report6_files/Report6_61_1.png)
    



```python
msno.heatmap(df_sales)
```




    <Axes: >




    
![png](Report6_files/Report6_62_1.png)
    


As we can see, missingness in `Day_Temp`, `No_of_Customers`, and `Sales` is strongly correlated. We have a reason to suspect **MAR**.


```python
msno.dendrogram(df_sales)
```




    <Axes: >




    
![png](Report6_files/Report6_64_1.png)
    


The sample size is very small and there are no clear patterns by looking at non-missing values. For us, this is clearly **MAR**. We do not want to remove the rows with missing values so we will apply MICE imputation.


```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

df_sales_clean = df_sales.copy()
mice_features = ['Day_Temp', 'No_of_Customers', 'Sales']

mice_imputer = IterativeImputer(random_state=1, max_iter=10)
mice_imputed = mice_imputer.fit_transform(df_sales[mice_features])

df_sales_clean[mice_features] = pd.DataFrame(mice_imputed, columns=mice_features)
df_sales_clean
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
      <th>Date</th>
      <th>Store_Type</th>
      <th>City_Type</th>
      <th>Day_Temp</th>
      <th>No_of_Customers</th>
      <th>Sales</th>
      <th>Product_Quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-10-01</td>
      <td>1</td>
      <td>1</td>
      <td>30.000000</td>
      <td>100.000000</td>
      <td>3112.000000</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-10-02</td>
      <td>2</td>
      <td>1</td>
      <td>32.000000</td>
      <td>115.000000</td>
      <td>3682.000000</td>
      <td>A</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-10-03</td>
      <td>3</td>
      <td>3</td>
      <td>31.000000</td>
      <td>104.267858</td>
      <td>2774.000000</td>
      <td>A</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-10-04</td>
      <td>1</td>
      <td>2</td>
      <td>29.000000</td>
      <td>105.000000</td>
      <td>3182.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-10-05</td>
      <td>1</td>
      <td>2</td>
      <td>33.000000</td>
      <td>104.000000</td>
      <td>1368.000000</td>
      <td>B</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2020-10-07</td>
      <td>2</td>
      <td>2</td>
      <td>27.965790</td>
      <td>99.926786</td>
      <td>2910.222481</td>
      <td>B</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2020-11-24</td>
      <td>2</td>
      <td>3</td>
      <td>26.000000</td>
      <td>90.000000</td>
      <td>4232.000000</td>
      <td>C</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2020-11-25</td>
      <td>3</td>
      <td>3</td>
      <td>26.657898</td>
      <td>96.000000</td>
      <td>2910.224806</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2020-11-26</td>
      <td>2</td>
      <td>2</td>
      <td>27.000000</td>
      <td>100.000000</td>
      <td>2356.000000</td>
      <td>B</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2020-11-28</td>
      <td>3</td>
      <td>1</td>
      <td>27.965790</td>
      <td>99.926786</td>
      <td>2910.222481</td>
      <td>A</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2020-11-29</td>
      <td>1</td>
      <td>1</td>
      <td>23.000000</td>
      <td>94.000000</td>
      <td>1254.000000</td>
      <td>A</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2020-11-30</td>
      <td>1</td>
      <td>1</td>
      <td>22.000000</td>
      <td>91.000000</td>
      <td>4232.000000</td>
      <td>A</td>
    </tr>
  </tbody>
</table>
</div>



We will also impute `Product_Quality` with the mode.


```python
product_quality_mode = df_sales['Product_Quality'].mode()
df_sales_clean['Product_Quality'] = df_sales_clean['Product_Quality'].fillna(product_quality_mode[0])
df_sales_clean
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
      <th>Date</th>
      <th>Store_Type</th>
      <th>City_Type</th>
      <th>Day_Temp</th>
      <th>No_of_Customers</th>
      <th>Sales</th>
      <th>Product_Quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-10-01</td>
      <td>1</td>
      <td>1</td>
      <td>30.000000</td>
      <td>100.000000</td>
      <td>3112.000000</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-10-02</td>
      <td>2</td>
      <td>1</td>
      <td>32.000000</td>
      <td>115.000000</td>
      <td>3682.000000</td>
      <td>A</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-10-03</td>
      <td>3</td>
      <td>3</td>
      <td>31.000000</td>
      <td>104.267858</td>
      <td>2774.000000</td>
      <td>A</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-10-04</td>
      <td>1</td>
      <td>2</td>
      <td>29.000000</td>
      <td>105.000000</td>
      <td>3182.000000</td>
      <td>A</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-10-05</td>
      <td>1</td>
      <td>2</td>
      <td>33.000000</td>
      <td>104.000000</td>
      <td>1368.000000</td>
      <td>B</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2020-10-07</td>
      <td>2</td>
      <td>2</td>
      <td>27.965790</td>
      <td>99.926786</td>
      <td>2910.222481</td>
      <td>B</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2020-11-24</td>
      <td>2</td>
      <td>3</td>
      <td>26.000000</td>
      <td>90.000000</td>
      <td>4232.000000</td>
      <td>C</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2020-11-25</td>
      <td>3</td>
      <td>3</td>
      <td>26.657898</td>
      <td>96.000000</td>
      <td>2910.224806</td>
      <td>A</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2020-11-26</td>
      <td>2</td>
      <td>2</td>
      <td>27.000000</td>
      <td>100.000000</td>
      <td>2356.000000</td>
      <td>B</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2020-11-28</td>
      <td>3</td>
      <td>1</td>
      <td>27.965790</td>
      <td>99.926786</td>
      <td>2910.222481</td>
      <td>A</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2020-11-29</td>
      <td>1</td>
      <td>1</td>
      <td>23.000000</td>
      <td>94.000000</td>
      <td>1254.000000</td>
      <td>A</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2020-11-30</td>
      <td>1</td>
      <td>1</td>
      <td>22.000000</td>
      <td>91.000000</td>
      <td>4232.000000</td>
      <td>A</td>
    </tr>
  </tbody>
</table>
</div>



These are discrete values so we cannot have floats, let's round them.


```python
float_cols = df_sales_clean.select_dtypes(include='float')
df_sales_clean[float_cols.columns] = df_sales_clean[float_cols.columns].round(0).astype('Int64')
df_sales_clean
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
      <th>Date</th>
      <th>Store_Type</th>
      <th>City_Type</th>
      <th>Day_Temp</th>
      <th>No_of_Customers</th>
      <th>Sales</th>
      <th>Product_Quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-10-01</td>
      <td>1</td>
      <td>1</td>
      <td>30</td>
      <td>100</td>
      <td>3112</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-10-02</td>
      <td>2</td>
      <td>1</td>
      <td>32</td>
      <td>115</td>
      <td>3682</td>
      <td>A</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-10-03</td>
      <td>3</td>
      <td>3</td>
      <td>31</td>
      <td>104</td>
      <td>2774</td>
      <td>A</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-10-04</td>
      <td>1</td>
      <td>2</td>
      <td>29</td>
      <td>105</td>
      <td>3182</td>
      <td>A</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-10-05</td>
      <td>1</td>
      <td>2</td>
      <td>33</td>
      <td>104</td>
      <td>1368</td>
      <td>B</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2020-10-07</td>
      <td>2</td>
      <td>2</td>
      <td>28</td>
      <td>100</td>
      <td>2910</td>
      <td>B</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2020-11-24</td>
      <td>2</td>
      <td>3</td>
      <td>26</td>
      <td>90</td>
      <td>4232</td>
      <td>C</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2020-11-25</td>
      <td>3</td>
      <td>3</td>
      <td>27</td>
      <td>96</td>
      <td>2910</td>
      <td>A</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2020-11-26</td>
      <td>2</td>
      <td>2</td>
      <td>27</td>
      <td>100</td>
      <td>2356</td>
      <td>B</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2020-11-28</td>
      <td>3</td>
      <td>1</td>
      <td>28</td>
      <td>100</td>
      <td>2910</td>
      <td>A</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2020-11-29</td>
      <td>1</td>
      <td>1</td>
      <td>23</td>
      <td>94</td>
      <td>1254</td>
      <td>A</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2020-11-30</td>
      <td>1</td>
      <td>1</td>
      <td>22</td>
      <td>91</td>
      <td>4232</td>
      <td>A</td>
    </tr>
  </tbody>
</table>
</div>




```python
corrs = df_sales_clean.corr(numeric_only=True)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a heatmap
sns.heatmap(corrs, 
            annot=True,         # Show correlation coefficients
            fmt=".2f",          # Format for coefficients
            cmap="coolwarm",    # Color palette
            vmin=-1, vmax=1,    # Fixed scale
            square=True,        # Make cells square
            linewidths=0.5,     # Line width between cells
            cbar_kws={"shrink": .75})  # Colorbar shrink

# Title and layout
plt.title("Correlation Heatmap", fontsize=16)
plt.tight_layout()

# Show plot
plt.show()
```


    
![png](Report6_files/Report6_71_0.png)
    



```python
corrs_kendall = df_sales_clean.corr(method='kendall', numeric_only=True)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a heatmap
sns.heatmap(corrs_kendall, 
            annot=True,         # Show correlation coefficients
            fmt=".2f",          # Format for coefficients
            cmap="coolwarm",    # Color palette
            vmin=-1, vmax=1,    # Fixed scale
            square=True,        # Make cells square
            linewidths=0.5,     # Line width between cells
            cbar_kws={"shrink": .75})  # Colorbar shrink

# Title and layout
plt.title("Correlation Heatmap", fontsize=16)
plt.tight_layout()

# Show plot
plt.show()
```


    
![png](Report6_files/Report6_72_0.png)
    



```python
# day temp vs Number of customers
sns.scatterplot(x=df_sales_clean['Day_Temp'], y=df_sales_clean['No_of_Customers'])
sns.regplot(x=df_sales_clean['Day_Temp'], y=df_sales_clean['No_of_Customers'], scatter=False, color='red')
cor = ss.pearsonr(df_sales_clean['Day_Temp'],df_sales_clean['No_of_Customers'])[0]
print("Correlation (Pearson): ", cor)
```

    Correlation (Pearson):  0.8179099116801952
    


    
![png](Report6_files/Report6_73_1.png)
    



```python
#Number of customers by Product Quality

mask = df_sales_clean['Product_Quality'].notna() & df_sales_clean['No_of_Customers'].notna()
spearman_corr, p_value = spearmanr(df_sales_clean.loc[mask, 'Product_Quality'], df_sales_clean.loc[mask, 'No_of_Customers'])

print(f"Spearman's rank correlation (Product_Quality vs No_of_Customers): {spearman_corr:.2f}, p-value: {p_value:.2f}")

sns.boxplot(x='Product_Quality', y='No_of_Customers', data=df_sales_clean)
plt.title("No_of_Customers by Product_Quality")
plt.show()
```

    Spearman's rank correlation (Product_Quality vs No_of_Customers): -0.21, p-value: 0.51
    


    
![png](Report6_files/Report6_74_1.png)
    



```python
import plotly.express as px

px.box(df_sales, x='Sales', facet_col='Store_Type')
```



As shown by the above distributions, we can see that `Sales` are generally correlated to `Store_Type`. 


```python
#Store type and city type (no missingness here)
SalesQ = pd.crosstab(df_sales_clean['Store_Type'], df_sales_clean['City_Type'])
SalesQm = pd.crosstab(df_sales_clean['Store_Type'], df_sales_clean['City_Type'], margins=True)
print(tabulate.tabulate(SalesQm, headers='keys', tablefmt='github'))
```

    | Store_Type   |   1 |   2 |   3 |   All |
    |--------------|-----|-----|-----|-------|
    | 1            |   3 |   2 |   0 |     5 |
    | 2            |   1 |   2 |   1 |     4 |
    | 3            |   1 |   0 |   2 |     3 |
    | All          |   5 |   4 |   3 |    12 |
    


```python
chi2sales, pq, dofq, expS = ss.chi2_contingency(SalesQ)

SalesExp = pd.DataFrame(expS, 
                           index=SalesQ.index, 
                           columns=SalesQ.columns)
print("Expected Table:")
print(tabulate.tabulate(SalesExp, headers='keys', tablefmt='github'))

print("Chi statistic for this data: ", chi2sales)
Vs = association(SalesQ, method="cramer")

print(f"Cramer's V for this data: {Vs:.2f}")
```

    Expected Table:
    |   Store_Type |       1 |       2 |    3 |
    |--------------|---------|---------|------|
    |            1 | 2.08333 | 1.66667 | 1.25 |
    |            2 | 1.66667 | 1.33333 | 1    |
    |            3 | 1.25    | 1       | 0.75 |
    Chi statistic for this data:  5.453333333333333
    Cramer's V for this data: 0.48
    

# Summary

There are many ways to *describe* our data:

- Measure **central tendency**.

- Measure its **variability**; **skewness** and **kurtosis**.

- Measure what **correlations** our data have.

All of these are **useful** and all of them are also **exploratory data analysis** (EDA).
