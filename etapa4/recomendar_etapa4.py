"""Gera o top-10 de um usuário com o modelo consolidado da etapa 4.

Uso: python recomendar_etapa4.py <pasta_modelo> <titulos.csv> [pseudonimo]
Sem pseudônimo, usa o mesmo usuário do exemplo determinístico da etapa 2.
Usuários sem histórico recebem a lista por popularidade (política de partida a frio documentada).
Jogos rejeitados explicitamente pelo usuário são bloqueados.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import load_npz

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding='utf-8')
    except Exception:
        pass

B = Path(sys.argv[1]); titulos = pd.read_csv(sys.argv[2], dtype={'item': str}).set_index('item').title.to_dict()
u = sys.argv[3] if len(sys.argv) > 3 else '000db80e6a62631f5c98'
meta = json.loads((B / 'modelo.json').read_text(encoding='utf-8'))
mp = json.loads((B / 'mapas.json').read_text(encoding='utf-8'))
X = load_npz(B / 'matriz_positivos.npz'); items = mp['items']; pop = np.asarray(X.sum(axis=0)).ravel()
idx = {x: i for i, x in enumerate(mp['users'])}
if u in idx:
    r = idx[u]
    if (B / 'pesos_item_item.npz').exists():
        s = (X[r] @ load_npz(B / 'pesos_item_item.npz')).toarray().ravel()
    else:
        s = np.load(B / 'fatores_usuarios.npy')[r] @ np.load(B / 'fatores_jogos.npy').T
    s = s.astype(float); s[X[r].indices] = -np.inf
    origem = f"{meta['familia']} {meta['hiperparametros']}"
    hist = [items[i] for i in X[r].indices]
else:
    s = pop.astype(float).copy(); origem = 'popularidade (usuário sem histórico)'; hist = []
for j in mp['rejeitados'].get(u, []):
    s[j] = -np.inf
rank = np.lexsort((np.arange(len(items)), -pop, -s))[:10]
print(f'Usuário: {u} | modelo: {origem}')
print('Histórico positivo:', '; '.join(titulos.get(i, i) for i in hist) or '—')
for k, i in enumerate(rank, 1):
    print(f'{k:2d}. {items[i]:>7} | {titulos.get(items[i], items[i])} | escore={s[i]:.6f}')
