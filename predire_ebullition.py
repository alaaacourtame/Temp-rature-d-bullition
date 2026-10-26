from pathlib import Path

import numpy as np
import joblib
import pandas as pd

from outils_descripteurs import preparer_mol, calculer

DOSSIER = Path(__file__).parent
donnees = joblib.load(DOSSIER / "modele_ebullition.joblib")
modele = donnees["modele"]
calibrateur = donnees["calibrateur"]
colonnes = donnees["colonnes"]

while True:
    smiles = input("\nEntrez le SMILES (vide pour quitter) : ").strip()
    if not smiles:
        break

    mol = preparer_mol(smiles)
    if mol is None:
        print("Erreur : SMILES invalide.")
        continue
        X = pd.DataFrame([calculer(mol)], columns=colonnes).replace([np.inf, -np.inf], np.nan)
        prediction = float(modele.predict(X)[0])
    
    if calibrateur is not None:
        prediction = float(calibrateur.predict([[prediction]])[0])
        
    print(f"\nTemperature d'ebullition estimee : {prediction:.2f} °C  (± {donnees['mae_test']:.1f} °C en moyenne)")
    print(f"Temperature d'ebullition estimee : {prediction + 273.15:.2f} K")
    print(f"Pression de reference : {donnees['pression_atm']} atm")
    
    if not donnees["y_min"] <= prediction <= donnees["y_max"]:
        print(f"Attention : hors de la plage d'entrainement ({donnees['y_min']:.0f} à {donnees['y_max']:.0f} °C), estimation peu fiable.")
   