"""Exporta o modelo final para a demonstração da oficina (página única, calculada no navegador).

Uso: python construir_demo.py <pasta_modelo_final> <titulos.csv> <resultados_etapa4> <pasta_demo>
Lê demo_corpo.html (modelo da página) em <pasta_demo> e grava:
  demo_publicacao.html  conteúdo com os dados embutidos (para publicação como página hospedada)
  index.html            arquivo completo; abre com dois cliques, sem servidor
Os pesos item-item são quantizados em 16 bits por linha; o script mede a fidelidade em relação ao modelo exato.
"""
import base64, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import load_npz

B, TIT, R, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
meta = json.loads((B / 'modelo.json').read_text(encoding='utf-8'))
mp = json.loads((B / 'mapas.json').read_text(encoding='utf-8'))
X = load_npz(B / 'matriz_positivos.npz').tocsr(); W = load_npz(B / 'pesos_item_item.npz').tocsr()
W.sort_indices(); items = mp['items']; N = len(items); assert N < 65536
pop = np.asarray(X.sum(axis=0)).ravel().astype(int)
tit = pd.read_csv(TIT, dtype=str).drop_duplicates('item').set_index('item').title.to_dict()
# nomes de oito jogos populares ausentes dos metadados da coleção, complementados manualmente apenas para exibição
COMPLEMENTO = {'72850': 'The Elder Scrolls V: Skyrim', '244850': 'Space Engineers', '219740': "Don't Starve",
               '238960': 'Path of Exile', '292030': 'The Witcher 3: Wild Hunt', '251570': '7 Days to Die',
               '12210': 'Grand Theft Auto IV', '381210': 'Dead by Daylight'}
titulos = [tit.get(i, COMPLEMENTO.get(i, f'App {i}')) for i in items]
rmax = np.zeros(N, dtype=np.float32); val = np.zeros(W.nnz, dtype=np.uint16)
for i in range(N):
    a, z = W.indptr[i], W.indptr[i + 1]
    if z > a:
        m = W.data[a:z].max(); rmax[i] = m
        val[a:z] = np.round(W.data[a:z] / m * 65535).astype(np.uint16)
b64 = lambda arr: base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()

# fidelidade: listas top-10 do modelo quantizado x modelo exato em 1.000 usuários
rng = np.random.default_rng(0); us = rng.choice(X.shape[0], 1000, replace=False); iguais = 0; comuns = 0
Wq = W.copy(); Wq.data = (val.astype(np.float64) / 65535) * np.repeat(rmax.astype(np.float64), np.diff(W.indptr))
idx = np.arange(N)
for u in us:
    h = X[u].indices
    a = np.asarray(X[u] @ W.toarray() if False else (X[u] @ W).todense()).ravel().astype(float); a[h] = -np.inf
    b = np.asarray((X[u] @ Wq).todense()).ravel().astype(float); b[h] = -np.inf
    la = np.lexsort((idx, -pop, -a))[:10]; lb = np.lexsort((idx, -pop, -b))[:10]
    iguais += int((la == lb).all()); comuns += len(set(la) & set(lb))
fid = {'listas_identicas': iguais / len(us), 'itens_em_comum': comuns / (10 * len(us))}

T = pd.read_csv(R / 'teste_modelos_finais.csv')
hp = meta['hiperparametros']
desc = f"{meta['familia']} (" + '; '.join(f"{ {'alpha': 'α', 'beta': 'β', 'k': 'k', 'lam': 'λ'}.get(k, k)} = {str(v).replace('.', ',')}" for k, v in hp.items()) + ')'
exemplo = [items.index(i) for i in ('620', '105600', '72850') if i in items] or list(np.argsort(-pop)[:3])
D = {'ids': [int(i) for i in items], 'titulos': titulos, 'pop': pop.tolist(), 'exemplo': [int(i) for i in exemplo],
     'w': {'ptr': b64(W.indptr.astype(np.uint32)), 'idx': b64(W.indices.astype(np.uint16)), 'val': b64(val), 'rmax': b64(rmax)},
     'meta': {'modelo': desc, 'interacoes': int(X.nnz), 'usuarios': int(X.shape[0]),
              'cob_pop': float(T[T.modelo == 'Popularidade'].cobertura.mean()),
              'cob_modelo': float(T[T.modelo == 'Modelo final (F)'].cobertura.mean())}}
corpo = (OUT / 'demo_corpo.html').read_text(encoding='utf-8')
assert '/*DADOS*/null/*FIM*/' in corpo
corpo = corpo.replace('/*DADOS*/null/*FIM*/', json.dumps(D, ensure_ascii=False, separators=(',', ':')))
(OUT / 'demo_publicacao.html').write_text(corpo, encoding='utf-8')
(OUT / 'index.html').write_text('<!doctype html>\n<html lang="pt-BR">\n<head>\n<meta charset="utf-8">\n'
                               '<meta name="viewport" content="width=device-width, initial-scale=1">\n</head>\n<body style="margin:0">\n'
                               + corpo + '\n</body>\n</html>\n', encoding='utf-8')
(OUT / 'fidelidade.json').write_text(json.dumps({**fid, 'nnz': int(W.nnz), 'bytes_pagina': len(corpo.encode())}, indent=1), encoding='utf-8')
print(json.dumps({**fid, 'nnz': int(W.nnz), 'kb': round(len(corpo.encode()) / 1024)}, ensure_ascii=False))
