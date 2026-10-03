"""Demonstração a partir do modelo persistido; não acessa perfis da Steam.
Uso: python recomendar.py [pseudonimo_do_usuario]
Sem argumento, usa o usuário do exemplo determinístico do relatório.
"""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
B=Path(__file__).resolve().parent;R=B/'resultados'
mp=json.loads((R/'mapas.json').read_text())
u=sys.argv[1] if len(sys.argv)>1 else json.loads((R/'exemplo.json').read_text())['user']
if u not in mp['users']:
    raise SystemExit('Usuário não está no modelo. O protótipo não cobre cold-start de usuário.')
idx=mp['users'].index(u);items=mp['items']
X=load_npz(R/'matriz_treino.npz');S=load_npz(R/'modelo_itemknn.npz')
pop=np.asarray(X.sum(axis=0)).ravel();scores=(X[idx]@S).toarray().ravel()
scores[X[idx].indices]=-np.inf
rank=np.lexsort((np.arange(len(items)),-pop,-scores))[:10]
titles=pd.read_csv(B/'dados/titulos.csv',dtype={'item':str}).set_index('item').title.to_dict()
for pos,i in enumerate(rank,1):
    print(f'{pos:2d}. {items[i]} | {titles.get(items[i],items[i])} | escore={scores[i]:.6f}')
