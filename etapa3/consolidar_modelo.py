"""Consolida o modelo escolhido para implantação e mede a latência de inferência.

Uso: python consolidar_modelo.py <pasta_dados> <pasta_resultados_etapa3> <pasta_saida> [A|B]

Regra de escolha (definida antes da leitura do teste): família com maior NDCG@10 médio de VALIDAÇÃO no ponto de
operação A do protocolo A; hiperparâmetros = configuração escolhida com mais frequência entre as sementes
(empate: semente 42). O modelo final é ajustado com todos os positivos, pois em operação todo o histórico é usado.
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import save_npz, csr_matrix
sys.path.insert(0, str(Path(__file__).resolve().parent))
import comum as c

D, RES, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
T = pd.read_csv(RES / 'A_teste.csv')
PONTO = sys.argv[4] if len(sys.argv) > 4 else 'A'
A0 = T[(T.ponto == 'A') & ~T.familia.isin(['Popularidade', 'Aleatório'])]
val = A0.groupby('familia').val_ndcg10.mean().sort_values(ascending=False)
fam = val.index[0]  # a família é sempre escolhida pelo ponto A; o ponto B apenas troca a configuração
A = T[(T.ponto == PONTO) & (T.familia == fam)]
cont = A[A.familia == fam].hp.value_counts()
top = cont[cont == cont.max()].index
hp_txt = top[0] if len(top) == 1 else A[(A.familia == fam) & (A.semente == 42)].hp.iloc[0]
hp = json.loads(hp_txt)

df, _ = c.carregar(D / 'interacoes.csv.gz')
pos = df[df.positive]
M = c.matriz(pos)
C = (M.X.T @ M.X).toarray()
t0 = time.perf_counter()
if fam == 'RP3β':
    mod = c.RP3beta(M, hp['alpha'], hp['beta'], hp['k']); W = mod.S
elif fam == 'EASE':
    mod = c.EASE(M, hp['lam'], C); W = csr_matrix(mod.B)
elif fam == 'ItemKNN':
    mod = c.ItemKNN(M, hp['k'], hp['shrink'], C); W = mod.S
elif fam == 'iALS':
    mod = c.IALS(M, hp['fatores'], hp['reg'], hp['alpha'], hp['iters']); W = None
else:
    raise SystemExit(f'família não prevista: {fam}')
t_fit = time.perf_counter() - t0
# latência: escores + ordenação top-10 para 1.000 usuários sorteados
rng = np.random.default_rng(0); us = rng.choice(M.X.shape[0], 1000, replace=False)
t0 = time.perf_counter()
S = mod.escores(us)
for n, u in enumerate(us):
    a = S[n].astype(float); a[M.X[u].indices] = -np.inf
    np.lexsort((np.arange(len(M.items)), -M.pop, -a))[:10]
lat = (time.perf_counter() - t0) / len(us) * 1000
save_npz(OUT / 'matriz_positivos.npz', M.X)
if W is not None:
    save_npz(OUT / 'pesos_item_item.npz', W)
else:
    np.save(OUT / 'fatores_usuarios.npy', mod.U); np.save(OUT / 'fatores_jogos.npy', mod.V)
rej = {}
for u, i in zip(df[~df.positive].user, df[~df.positive].item):
    if i in M.im:
        rej.setdefault(u, []).append(M.im[i])
meta = {'familia': fam, 'ponto': PONTO, 'hiperparametros': hp, 'val_ndcg_medio_por_familia': val.to_dict(),
        'frequencia_hp': cont.to_dict(), 'usuarios': len(M.users), 'jogos': len(M.items),
        'interacoes': int(M.X.nnz), 'segundos_ajuste': t_fit, 'ms_por_usuario_top10': lat,
        'nnz_pesos': int(W.nnz) if W is not None else None}
(OUT / 'modelo.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
(OUT / 'mapas.json').write_text(json.dumps({'users': M.users, 'items': M.items, 'rejeitados': rej}), encoding='utf-8')
print(json.dumps(meta, indent=1, ensure_ascii=False))
