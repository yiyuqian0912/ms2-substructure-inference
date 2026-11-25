"""Keep the exported molecular graph separate from its reported ion composition."""
from collections import Counter
from functools import lru_cache
import hashlib
import json
import re

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog('rdApp.warning')
ELEMENTS=('C','H','N','O','P','S','F','Cl','Br','I','B','Si','Se')
ATOMS=tuple(e for e in ELEMENTS if e!='H')
ATOM_DIM=len(ATOMS)+5
BOND_DIM=7
NUMERIC_DIM=len(ELEMENTS)+2


def composition(formula):
    pairs=re.findall(r'([A-Z][a-z]?)(\d*)',formula)
    if ''.join(e+n for e,n in pairs)!=formula or not pairs:
        raise ValueError(f'Unsupported formula notation: {formula}')
    result=Counter()
    for element,count in pairs:
        if element not in ELEMENTS:raise ValueError(f'Unsupported element: {element}')
        result[element]+=int(count or 1)
    return result


def ion_sign(ion):
    if ion not in ('[M+H]+','[M-H]-'):raise ValueError(f'Unsupported ion: {ion}')
    return 1 if ion=='[M+H]+' else -1


@lru_cache(maxsize=150000)
def structure(smiles,formula,ion):
    ion_sign(ion)
    mol=Chem.MolFromSmiles(smiles)
    if mol is None:raise ValueError('Invalid exported SMILES')
    mol=Chem.RemoveHs(mol);Chem.RemoveStereochemistry(mol)
    if not mol.GetNumAtoms():raise ValueError('Empty fragment graph')
    graph_formula=rdMolDescriptors.CalcMolFormula(mol)
    expected=composition(formula)
    actual=Counter(a.GetSymbol() for a in mol.GetAtoms())
    if any(actual[e]!=expected[e] for e in ELEMENTS if e!='H'):
        raise ValueError('Heavy-atom composition disagrees with exported graph')
    canonical=Chem.MolToSmiles(mol,isomericSmiles=False)
    key=hashlib.sha256(json.dumps([canonical,formula,ion],separators=(',',':')).encode()).hexdigest()
    return dict(key=key,smiles=canonical,formula=formula,ion=ion,
                smiles_formula=graph_formula,formula_smiles_consistent=graph_formula==formula)


def graph_features(label):
    mol=Chem.MolFromSmiles(label['smiles'])
    if mol is None:raise ValueError('Invalid catalog graph')
    atoms=[]
    for atom in mol.GetAtoms():
        symbol=atom.GetSymbol()
        atoms.append([float(symbol==e) for e in ATOMS]+[float(symbol not in ATOMS),
            float(atom.GetIsAromatic()),float(atom.IsInRing()),atom.GetFormalCharge()/3.,atom.GetDegree()/4.])
    edges=[];bonds=[]
    types=(Chem.BondType.SINGLE,Chem.BondType.DOUBLE,Chem.BondType.TRIPLE,Chem.BondType.AROMATIC)
    for bond in mol.GetBonds():
        feature=[float(bond.GetBondType()==t) for t in types]+[float(bond.GetBondType() not in types),
                     float(bond.GetIsConjugated()),float(bond.IsInRing())]
        i,j=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        edges.extend([(i,j),(j,i)]);bonds.extend([feature,feature])
    comp=composition(label['formula'])
    numeric=[comp[e]/50. for e in ELEMENTS]+[label['mz']/1000.,float(ion_sign(label['ion']))]
    return (torch.tensor(atoms,dtype=torch.float32),torch.tensor(edges,dtype=torch.long).reshape(-1,2).T,
            torch.tensor(bonds,dtype=torch.float32).reshape(-1,BOND_DIM),torch.tensor(numeric,dtype=torch.float32))
