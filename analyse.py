import pandas as pd
import matplotlib.pyplot as plt


# IMPORTATION ET AFFICHAGE 

df = pd.read_csv("result.csv", sep=";")

print("Premieres lignes:")
print(df.head())

print("\nTypes de colonnes:")
print(df.dtypes)

print("\nValeurs manquantes:")
print(df.isna().sum())

print("Nb lignes:", df.shape[0])
print("Nb colonnes :", df.shape[1])



# Affichage de la ligne football pour comprendre pourquoi elle etait manquante apres le nettoyage
# print(df[df["football_interest_score"].isna()])


# TYPES DES COLONNES

# Corrections des types 
df["gaming_interest_score"] = df["gaming_interest_score"].astype("float32")
df["insta_design_interest_score"] = df["insta_design_interest_score"].astype("float32")
df["football_interest_score"] = df["football_interest_score"].astype("float32")

df["age"] = pd.to_numeric(df["age"], errors="coerce")

df["campaign_success"] = df["campaign_success"].astype(str).str.strip()
df["campaign_success"] = df["campaign_success"].map({"True": True, "False": False})

print("\nApres correction:")
print(df.dtypes)


# VALEURS MANQUANTES

# suppression des deux champs manquants --> ce qui a causé la disparition de la ligne avec football_interest_score manquant
df = df.dropna(subset=["age"])
df = df.dropna(subset=["recommended_product"])

#re rappel de la taille du dataset 
print("Nb lignes:", df.shape[0])
print("Nb colonnes :", df.shape[1])

#verification qu'il y a plus de valeurs manquantes
print(df.isna().sum())


# NETTOYAGE

df["canal_recommande"] = df["canal_recommande"].astype(str).str.strip().str.lower()
df["recommended_product"] = df["recommended_product"].astype(str).str.strip().str.lower()

print("\nExemples de valeurs pour canal_recommande :")
print(df["canal_recommande"].value_counts().head())

print("\nExemples de valeurs pour recommended_product :")
print(df["recommended_product"].value_counts().head())

#Vu l'apparision de test et non_defini, je les supprime ici
df = df[df["recommended_product"] != "test"]
df = df[df["canal_recommande"] != "non_defini"]

#re rappel de la taille du dataset 
print("\nNb lignes:", df.shape[0])
print("Nb colonnes :", df.shape[1])



# VERIFICATION NETTOYAGE

print("\nTypes de colonnes FINAUX :")
print(df.dtypes)

print("\nValeurs manquantes FINALES :")
print(df.isna().sum())





# ANOMALIES

#premiere verification sur les chiffres qu'on peut avoir
print("\nStatistiques descriptives après premier nettoryage")
print(df.describe())

#deuximeme verification des min et max sur le score et l'age
print("\nValeurs minimales et maximales des scores et de l'âge :")
for col in ["gaming_interest_score", "insta_design_interest_score", "football_interest_score", "age"]:
    print(col, "  min:", df[col].min(), "/ max:", df[col].max())

df = df[df["age"] >= 20]


anomalies_scores = df[
    (df["gaming_interest_score"] < 0) | (df["gaming_interest_score"] > 100) |
    (df["insta_design_interest_score"] < 0) | (df["insta_design_interest_score"] > 100) |
    (df["football_interest_score"] < 0) | (df["football_interest_score"] > 100)
]

print("\nNb lignes avec score pas normal) :", len(anomalies_scores))
print("\nExemples:")
print(anomalies_scores.head())


# Nouveau dataframe sans anomalie et nettoyé
df_clean = df.drop(anomalies_scores.index)

print("\nVerification dans le nouveau dataframe :")
for col in ["gaming_interest_score", "insta_design_interest_score", "football_interest_score", "age"]:
    print(col, "  min:", df_clean[col].min(), "/ max:", df_clean[col].max())





# KPI ET VISUALISATIONS

#re rappel de la taille du dataset 
print("\nNb lignes:", df_clean.shape[0])
print("Nb colonnes :", df_clean.shape[1])


# KPI 1: Le taux de reussite globale
success_rate_global = df_clean["campaign_success"].mean()  
taux_pourcent = round(success_rate_global * 100, 2)
print("Taux de réussite global de la campagne :", taux_pourcent, "%")



# KPI 2 : Taux de réussite par produit recommandé
success_by_product = df_clean.groupby("recommended_product")["campaign_success"].mean().sort_values(ascending=False)
print("\nTaux de réussite par produit recommandé (en %) :")
print((success_by_product * 100).round(2))


# Graphique en barres 
plt.figure(figsize=(8, 4))
(success_by_product * 100).plot(kind="bar", color="skyblue")
plt.title("Taux de réussite par produit recommandé")
plt.ylabel("Taux de réussite (%)")
plt.xlabel("Produit recommandé")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()



# KPI 3 : Taux de reussite par canal '
success_by_channel = df_clean.groupby("canal_recommande")["campaign_success"].mean().sort_values(ascending=False)

print("\nTaux de reussite par canal:")
print((success_by_channel * 100).round(2))

plt.figure(figsize=(8, 4))
(success_by_channel * 100).plot(kind="bar", color="orange")
plt.title("Taux de réussite par canal recommandé")
plt.ylabel("Taux de réussite (%)")
plt.xlabel("Canal recommandé")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()


# KPI 4 : Taux de reussite par tranche d'age
bins = [20, 24, 34, 44, 60]
labels = ["20-24", "25-34", "35-44", "45-60"]


df_clean["age_group"] = pd.cut(df_clean["age"], bins=bins, labels=labels, right=True, include_lowest=True)

success_by_age_group = df_clean.groupby("age_group")["campaign_success"].mean()
print("\nTaux de réussite par tranche d'âge (en %) :")
print((success_by_age_group * 100).round(2))

plt.figure(figsize=(8, 4))
(success_by_age_group * 100).plot(kind="bar", color="green")
plt.title("Taux de réussite par tranche d'âge")
plt.ylabel("Taux de réussite (%)")
plt.xlabel("Tranche d'âge")
plt.xticks(rotation=0)
plt.tight_layout()



print("\nMATRICE DE CORRÉLATION")

cols_corr = [
    "gaming_interest_score",
    "insta_design_interest_score",
    "football_interest_score",
    "age",
    "campaign_success"
]

corr = df_clean[cols_corr].corr()
print("\nMatrice de corrélation :")
print(corr)

# GRAPHIQUE CORRELATION

# 1) age vs interet pour le gaming
plt.figure(figsize=(6, 4))
plt.scatter(df_clean["age"], df_clean["gaming_interest_score"], alpha=0.5)
plt.title("Âge vs score d'intérêt pour le gaming")
plt.xlabel("Âge")
plt.ylabel("Score gaming")
plt.grid(True)

# 2) age vs interet pour le football
plt.figure(figsize=(6, 4))
plt.scatter(df_clean["age"], df_clean["football_interest_score"], alpha=0.5, color="orange")
plt.title("Âge vs score d'intérêt pour le football")
plt.xlabel("Âge")
plt.ylabel("Score football")
plt.grid(True)


plt.show()




