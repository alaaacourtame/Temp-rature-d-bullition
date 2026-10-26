import numpy as np
import pandas as pd
import joblib

from sklearn.base import clone
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from outils_descripteurs import COLONNES, preparer_mol, smiles_canonique, calculer

# ---------- RÉGLAGES À VÉRIFIER ----------
UNITE_DONNEES = "C"      # unité de la colonne BP dans BP.csv : "C" ou "K"
SEUIL_CONFLIT = 15.0     # écart-type (°C) au-delà duquel un doublon est jugé incohérent
# -----------------------------------------

# 1. Charger et nettoyer les données
df = pd.read_csv("BP.csv").dropna(subset=["SMILES", "BP"])
df["BP"] = pd.to_numeric(df["BP"], errors="coerce")
df = df.dropna(subset=["BP"])
if UNITE_DONNEES == "K":
    df["BP"] = df["BP"] - 273.15

print("BP (°C) : min %.1f | médiane %.1f | max %.1f" % (df.BP.min(), df.BP.median(), df.BP.max()))

# SMILES canoniques : évite que la même molécule soit à la fois en train ET en test
df["SMILES_can"] = df["SMILES"].map(smiles_canonique)
df = df.dropna(subset=["SMILES_can"])

# Doublons : médiane des valeurs (au lieu de garder la 1re ligne), on écarte les incohérents
if "Subset" not in df.columns:
    df["Subset"] = "Train"
g = df.groupby("SMILES_can").agg(
    median=("BP", "median"), std=("BP", "std"), subset=("Subset", "first")
).reset_index()
conflits = g["std"].fillna(0) > SEUIL_CONFLIT
print(f"Molécules uniques : {len(g)} | doublons incohérents écartés : {conflits.sum()}")
g = g[~conflits].reset_index(drop=True)

# 2. Descripteurs
X = pd.DataFrame(
    [calculer(preparer_mol(s)) for s in g["SMILES_can"]], columns=COLONNES
).replace([np.inf, -np.inf], np.nan)
y = g["median"].to_numpy(dtype=float)

# 3. Découpage : on respecte la colonne Subset du fichier (Train / Validation / Blind)
sub = g["subset"].to_numpy()
if {"Validation", "Blind"} <= set(sub):
    m_tr, m_val, m_bl = sub == "Train", sub == "Validation", sub == "Blind"
    X_train, y_train = X[m_tr], y[m_tr]
    X_test, y_test = X[m_val], y[m_val]          # sert à choisir (calibration)
    X_blind, y_blind = X[m_bl], y[m_bl]          # jamais utilisé pour décider : verdict final
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    X_blind, y_blind = None, None
print(f"Train : {len(y_train)} | Validation : {len(y_test)} | Blind : {0 if y_blind is None else len(y_blind)}")

# 4. Modèles et paramètres
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("var", VarianceThreshold(0.0)),
    ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
])

grille = {
    "model__n_estimators": [300, 500, 800],
    "model__max_depth": [None, 20, 30, 40],
    "model__min_samples_leaf": [1, 2, 3],
    "model__max_features": [0.7, 1.0],
}
parametres = [
    {"model": [RandomForestRegressor(random_state=42, n_jobs=-1)], **grille},
    {"model": [ExtraTreesRegressor(random_state=42, n_jobs=-1)], **grille},
]

# 5. Recherche des meilleurs paramètres (uniquement sur le train)
recherche = RandomizedSearchCV(
    pipeline, param_distributions=parametres, n_iter=30,
    scoring="neg_mean_absolute_error", cv=5, random_state=42, n_jobs=-1, verbose=1,
)
recherche.fit(X_train, y_train)
meilleur = recherche.best_estimator_

# 6. Correction du biais de régression vers la moyenne
#    Les forêts "rétrécissent" les prédictions : trop bas pour les hauts Teb, trop haut pour les bas.
#    On apprend une droite de correction sur des prédictions out-of-fold (sans fuite de données).
oof = cross_val_predict(clone(meilleur), X_train, y_train, cv=5, n_jobs=-1)
calibrateur = LinearRegression().fit(oof.reshape(-1, 1), y_train)

pred_brut = meilleur.predict(X_test)
pred_cal = calibrateur.predict(pred_brut.reshape(-1, 1))

mae_brut = mean_absolute_error(y_test, pred_brut)
mae_cal = mean_absolute_error(y_test, pred_cal)
utiliser_calibration = mae_cal < mae_brut
pred_test = pred_cal if utiliser_calibration else pred_brut
print(f"\nMAE test sans calibration : {mae_brut:.2f} °C | avec : {mae_cal:.2f} °C "
      f"-> calibration {'ACTIVÉE' if utiliser_calibration else 'désactivée'}")

# 7. Évaluation honnête sur le test
mae = mean_absolute_error(y_test, pred_test)
rmse = np.sqrt(mean_squared_error(y_test, pred_test))
r2 = r2_score(y_test, pred_test)
biais = float(np.mean(pred_test - y_test))
p10 = np.mean(np.abs(y_test - pred_test) <= 10) * 100

print("\nMeilleurs paramètres :", recherche.best_params_)
print(f"MAE   : {mae:.2f} °C")
print(f"RMSE  : {rmse:.2f} °C")
print(f"R²    : {r2:.4f}")
print(f"Biais moyen (prédit - réel) : {biais:+.2f} °C")
print(f"Prédictions avec erreur <= 10 °C : {p10:.1f} %")

tranches = pd.qcut(pd.Series(y_test), 5, duplicates="drop")
detail = pd.DataFrame({"tranche": tranches, "biais": pred_test - y_test, "abs": np.abs(pred_test - y_test)})
print("\nBiais et MAE par tranche de Teb réelle :")
print(detail.groupby("tranche", observed=True).agg(biais=("biais", "mean"), MAE=("abs", "mean")).round(2))

if y_blind is not None:
    pb = meilleur.predict(X_blind)
    if utiliser_calibration:
        pb = calibrateur.predict(pb.reshape(-1, 1))
    print("\n--- Jeu BLIND (verdict final) ---")
    print(f"MAE : {mean_absolute_error(y_blind, pb):.2f} °C | "
          f"RMSE : {np.sqrt(mean_squared_error(y_blind, pb)):.2f} °C | "
          f"R² : {r2_score(y_blind, pb):.4f} | biais : {np.mean(pb - y_blind):+.2f} °C")

tranches_p = pd.qcut(pd.Series(pred_test), 5, duplicates="drop")
detail_p = pd.DataFrame({"tranche": tranches_p, "biais": pred_test - y_test})
print("\nBiais par tranche de Teb PRÉDITE (c'est ce que tu vois à l'usage) :")
print(detail_p.groupby("tranche", observed=True)["biais"].mean().round(2))

# 8. Réentraîner sur TOUTES les données pour le modèle final
modele_final = clone(meilleur).fit(X, y)
calibrateur_final = None
if utiliser_calibration:
    oof_all = cross_val_predict(clone(meilleur), X, y, cv=5, n_jobs=-1)
    calibrateur_final = LinearRegression().fit(oof_all.reshape(-1, 1), y)

# 9. Sauvegarde
joblib.dump({
    "modele": modele_final,
    "calibrateur": calibrateur_final,
    "colonnes": COLONNES,
    "unite": "C",
    "pression_atm": 1.0,
    "mae_test": mae,
    "r2_test": r2,
    "biais_test": biais,
    "y_min": float(y.min()),
    "y_max": float(y.max()),
}, "modele_ebullition.joblib")

print("\nModèle sauvegardé dans modele_ebullition.joblib")