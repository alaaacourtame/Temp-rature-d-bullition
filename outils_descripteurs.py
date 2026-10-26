"""Code partagé entre l'entraînement et la prédiction (garantit des descripteurs identiques)."""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

EXCLUS = {"Ipc", "AvgIpc"}
DESCRIPTEURS = [(n, f) for n, f in Descriptors._descList if n not in EXCLUS]
COLONNES = [n for n, _ in DESCRIPTEURS]
_chooser = rdMolStandardize.LargestFragmentChooser()


def preparer_mol(smiles):
    """SMILES -> Mol, en gardant le plus gros fragment (retire sels / solvants)."""
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None
    try:
        mol = _chooser.choose(mol)
    except Exception:
        pass
    return mol


def smiles_canonique(smiles):
    mol = preparer_mol(smiles)
    return None if mol is None else Chem.MolToSmiles(mol)


def calculer(mol):
    """Liste des descripteurs ; NaN si erreur, inf ou valeur aberrante (>1e9)."""
    valeurs = []
    for _, fonction in DESCRIPTEURS:
        try:
            v = float(fonction(mol))
        except Exception:
            v = np.nan
        if not np.isfinite(v) or abs(v) > 1e9:
            v = np.nan
        valeurs.append(v)
    return valeurs