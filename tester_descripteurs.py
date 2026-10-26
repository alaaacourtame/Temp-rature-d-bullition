
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

fichier = Path(__file__).parent / "BP.csv"
df = pd.read_csv(fichier)
df.columns = df.columns.str.strip()

descripteurs = [
    (nom, fonction)
    for nom, fonction in Descriptors._descList
    if nom != "AvgIpc"
]
smiles = str(df.iloc[0]["SMILES"])
mol = Chem.MolFromSmiles(smiles)

print("Première molécule :", smiles, flush=True)
print("Nombre de descripteurs :", len(descripteurs), flush=True)

for i, (nom, fonction) in enumerate(descripteurs, start=1):
    print(f"Descripteur {i}/ {len(descripteurs)} : {nom}", flush=True)

    try:
        valeur = fonction(mol)
        print("  Résultat :", valeur, flush=True)
    except Exception as erreur:
        print("  Erreur :", erreur, flush=True)

print("TEST TERMINE", flush=True)
