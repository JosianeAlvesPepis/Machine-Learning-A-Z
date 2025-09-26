# Importance of Training-Test Split in ML Model Evaluation

When building a Machine Learning model, it's essential to split the dataset into **two parts**:
- **Training set (usually ~80%)** -> used to train the model (e.g., fitting a Linear Regression to predict car prices).
- **Test set (usually ~20%)** -> used to evaluate the model's performance on unseen data.

By separating the data:
- The model learns patterns from the  **training set**.
- We then check if the model can **generalize** by testing it on the **test set**.
- Since we already know the true values in the test set, we can compare them to the predictions and calculate performance metrics.

This process helps us identify if the model is **accurate enough** or if it needs improvements (e.g., feature engineering, parameter tuning, or trying another algorithm).

---

# Feature Scaling

Feature scaling is a technique to bring numerical features to a similar scale.
It's applied **to columns (features)**, not to the dataset as a whole.
This ensures that no variable dominates the model simply because it has larger values.

## Common Types

### 1. Normalization (Min-Max Scaling)

**Formula:**
$$x_{normalized} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

- Transform all values in a column to the **range [0, 1]**
- Useful when you don't know the distribution of your data.
- Example: scaling salaries so that the lowest salary becomes 0 and the highest becomes 1.

### 2. Standardization (Z-score Scaling)

**Formula:**
$$z = \frac{x - \mu}{\sigma}$$

- Substracts the **mean** $\mu$ and divides by the **standard deviation** $\sigma$.
- Results in data with **mean 0** and **standart deviation 1**.
- Most values fall in the range [3, -3], but outliers may fall outside.
- Commonly used in algorithms that assume normally distributed data (e.g., Logistic Regression, SVM).

### Why It Matters

Imagine you have a dataset with **Age** (e.g., 20-60 years) and **Salary** (e.g., 30,000-150,000).
If we try to compare similarity between people, the large salary values will dominate the smaller age values.

By applying **feature scaling**, both features are transformed to comparable ranges, making it easier for algorithms (like distance-based models: KNN, clustering) to work properly.

___

# Columns in Machine Learning

## Categorical Features (Independent Variables)
- Columns with categories (non-numeric).
- Examples: `Country`, `Gender`, `Product Type`.
- Need to be encoded (e.g., **OneHotEncoder**, **LabelEnconder**) so that algorithms can understand them.

## Features (Independent Variables / Predictors)
- The **input columns** we use to make predictions.
- Can be numerical or categorical (after encoding).
- Examples: if predicting **Car Price**, the features could be `Age`, `Mileage`, `Engine Size`.

## Target (Dependent Variable / Labels)
- The **output column** we want to predict.
- Example: 
    - For **regression**: `Car Price` (continuous value)
    - For **classification**: `Spam` (Yes/No), `Customer Segment` (A/B/C).

---

# Using `.iloc` in Pandas

`.iloc` is a method in pandas used to select data by **row and column positions (index numbers)**, not by names.
Think of it as "index-location"

## Basic Syntax

```
    dataset.iloc[rows, columns]

```
- `rows`: index positions of the rows you want.
- `columns`: index positions of the columns you want.

## Why It's Useful
- Makes it easy to **separate features (x)** and **Target (y)**.
- Works consistently even if column names change.
- Very common in preprocessing steps.

___


