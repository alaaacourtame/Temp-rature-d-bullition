
import faulthandler
faulthandler.enable()

from rdkit import Chem
from rdkit.Chem import Descriptors

mol = Chem.MolFromSmiles("C=O")

print("Début du diagnostic", flush=True)

for nom, fonction in Descriptors._descList:
    if nom == "AvgIpc":
        continue

    print(f"Test : {nom}", flush=True)
    resultat = fonction(mol)
    print(f"OK : {resultat}", flush=True)

print("Tous les tests sont terminés.", flush=True)
