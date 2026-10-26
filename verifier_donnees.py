
from pathlib import Path
import pandas as pd
from rdkit import Chem

fichier = Path(__file__).parent / "BP.csv"

if not fichier.exists():
    raise FileNotFoundError("Le fichier BP.csv est absent du dossier.")

df = pd.read_csv(fichier)
df.columns = df.columns.str.strip()

print("Nombre de lignes :", len(df))
print("Colonnes disponibles :")
print(df.columns.tolist())

colonnes = ["NAME", "SMILES", "BP"]
manquantes = [c for c in colonnes if c not in df.columns]

if manquantes:
    raise ValueError(f"Colonnes manquantes : {manquantes}")

print("\nExemples de molécules :")
print(df[colonnes].head(10).to_string(index=False))

df["BP"] = pd.to_numeric(df["BP"], errors="coerce")

print("\nStatistiques de BP en °C :")
print(df["BP"].describe())

valides = df["SMILES"].dropna().apply(
    lambda s: Chem.MolFromSmiles(str(s)) is not None
)

print("\nStructures SMILES valides :", int(valides.sum()))
print("Valeurs BP manquantes :", int(df["BP"].isna().sum()))
