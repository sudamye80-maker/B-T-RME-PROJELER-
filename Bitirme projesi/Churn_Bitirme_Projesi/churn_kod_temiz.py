import sqlite3
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score

warnings.filterwarnings("ignore")
np.random.seed(42)

n = 1000
df_ham = pd.DataFrame({
    "Uyelik_Ayi": np.random.randint(1, 60, n),
    "Aylik_Ucret": np.round(np.random.uniform(50, 500, n), 2),
    "Destek_Cagrisi": np.random.poisson(1.8, n),
    "Sozlesme": np.random.choice(["Aylik", "1_Yillik", "2_Yillik"], n, p=[0.5, 0.3, 0.2])
})
risk = (
    (df_ham["Sozlesme"] == "Aylik") * 1.5 +
    (df_ham["Destek_Cagrisi"] >= 3) * 1.8 -
    (df_ham["Uyelik_Ayi"] > 24) * 1.2 +
    np.random.normal(0, 0.5, n)
)

df_ham["Churn"] = (risk > 1.5).astype(int)
df_ham.loc[0:25, "Aylik_Ucret"] = np.nan
conn = sqlite3.connect("musteri.db")
df_ham.to_sql("musteriler", conn, if_exists="replace", index=False)
df = pd.read_sql_query("SELECT * FROM musteriler", conn)
conn.close()
df["Aylik_Ucret"] = df["Aylik_Ucret"].fillna(df["Aylik_Ucret"].median())
df["Aylik_Ucret"] = np.clip(df["Aylik_Ucret"], None, df["Aylik_Ucret"].quantile(0.95))
df["Cagri_Yogunlugu"] = df["Destek_Cagrisi"] / (df["Uyelik_Ayi"] + 1)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.countplot(data=df, x="Sozlesme", hue="Churn", ax=ax[0]).set_title("Sözleşme vs Kayıp")
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm", ax=ax[1]).set_title("Korelasyon")
plt.tight_layout()
plt.show()
df = pd.get_dummies(df, columns=["Sozlesme"], drop_first=True)
X, y = df.drop("Churn", axis=1), df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

parametreler = dict(n_estimators=[50, 100], max_depth=[4, 8])
grid = GridSearchCV(RandomForestClassifier(random_state=42), parametreler, cv=StratifiedKFold(5), scoring="f1")
grid.fit(X_train, y_train)

model = grid.best_estimator_
y_pred = model.predict(X_test)

print(f"F1-Score: {f1_score(y_test, y_pred):.2f} | ROC-AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.2f}")
print(classification_report(y_test, y_pred))

pd.Series(model.feature_importances_, index=X.columns).sort_values().plot(kind="barh", title="Özellik Önem Düzeyleri")
plt.tight_layout()
plt.show()
