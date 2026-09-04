import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sns.set(style="whitegrid")

# Dataset sintético com a mesma estrutura do MKT.csv original
# (171 amostras, colunas: youtube, facebook, newspaper, sales)
rng = np.random.default_rng(42)
n = 171
youtube   = rng.uniform(0.7,  449.0, n)
facebook  = rng.uniform(0.0,  300.0, n)
newspaper = rng.uniform(0.3,  114.0, n)
sales     = (
    3.0
    + 0.045 * youtube
    + 0.188 * facebook
    + 0.001 * newspaper
    + rng.normal(0, 1.5, n)
).clip(1.92, 32.4)

data = pd.DataFrame({
    "youtube":   youtube,
    "facebook":  facebook,
    "newspaper": newspaper,
    "sales":     sales,
})

# Para usar o CSV original quando tiver: descomente a linha abaixo
# data = pd.read_csv("MKT.csv")

print(data.head())
data.info()

# Verificação de qualidade
print("\nMissing values:")
print(data.isnull().sum())
print("\nDuplicates:", data.duplicated().sum())
print("\nDescriptive statistics:")
print(data.describe())

# Histogramas
data.hist(bins=30, figsize=(12, 8), color="skyblue", edgecolor="black")
plt.suptitle("Variable Distribution Histograms")
plt.tight_layout()
plt.show()

# Heatmap de correlação
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation HeatMap between Variables")
plt.show()

# Scatterplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, feature in zip(axes, ["youtube", "facebook", "newspaper"]):
    sns.scatterplot(data=data, x=feature, y="sales", ax=ax)
    ax.set_title(f"{feature.capitalize()} vs Sales")
    ax.set_xlabel(f"Investment in {feature.capitalize()}")
    ax.set_ylabel("Sales")
plt.tight_layout()
plt.show()

# Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, palette="Set2")
plt.title("Boxplot for Outlier Detection")
plt.show()

# Modelo de regressão linear
X = data[["youtube", "facebook", "newspaper"]]
y = data["sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nMean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:         ", r2_score(y_test, y_pred))

print("\nModel coefficients:")
print(pd.DataFrame(model.coef_, X.columns, columns=["Coefficient"]))


def forecast_sales(youtube: float, facebook: float, newspaper: float) -> float:
    inv = pd.DataFrame([[youtube, facebook, newspaper]], columns=["youtube", "facebook", "newspaper"])
    return model.predict(inv)[0]


print("\nSales Forecast — $100 Youtube / $150 Facebook / $50 Newspaper:")
print(f"  ${forecast_sales(100, 150, 50):.2f}")

print("\nSales Forecast — $200 Youtube / $200 Facebook / $100 Newspaper:")
print(f"  ${forecast_sales(200, 200, 100):.2f}")

print("\nSales Forecast — $150 Youtube / $100 Facebook / $50 Newspaper:")
print(f"  ${forecast_sales(150, 100, 50):.2f}")
