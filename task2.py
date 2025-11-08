import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("titanic dataset.txt")

print(df.info())
print(df.describe())
print(df.head())

print("\nMissing values:\n", df.isnull().sum())

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.drop('Cabin', axis=1)

df['Sex'] = df['Sex'].map({'male':0, 'female':1})
df['Embarked'] = df['Embarked'].map({'S':0, 'C':1, 'Q':2})

sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival Count by Sex")
plt.show()

sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival Count by Pclass")
plt.show()

sns.histplot(data=df, x='Age', hue='Survived', multiple='stack', bins=10)
plt.title("Age Distribution by Survival")
plt.show()

numeric_cols = df.select_dtypes(include=['int64', 'float64'])
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
