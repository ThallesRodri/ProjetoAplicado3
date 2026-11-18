"""Reprodução independente da prova de conceito da etapa 2 (semente 42)."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import comum as c

D = Path(sys.argv[1]); R = Path(sys.argv[2]); R.mkdir(parents=True, exist_ok=True)
df, aud = c.carregar(D / 'interacoes.csv.gz')
pos = df[df.positive]
P = c.particao_aleatoria(pos, 42)
M = c.matriz(P.treino)
C = (M.X.T @ M.X).toarray()
grade = []
t0 = time.perf_counter()
for k in (20, 50, 100):
    for s in (0, 10, 50):
        m, _ = c.avaliar(c.ItemKNN(M, k, s, C), M, P.validacao)
        grade.append({'k': k, 'shrink': s, **m})
g = pd.DataFrame(grade).sort_values(['ndcg10', 'k', 'shrink'], ascending=[False, True, True])
b = g.iloc[0]
knn = c.ItemKNN(M, int(b.k), int(b.shrink), C)
mp, pp = c.avaliar(c.Popularidade(M), M, P.teste)
mk, pk = c.avaliar(knn, M, P.teste)
mv, _ = c.avaliar(c.Popularidade(M), M, P.validacao)
ic = c.bootstrap_pareado(pk.ndcg10, pp.ndcg10)
out = {'auditoria': aud, 'treino': len(P.treino), 'usuarios_treino': len(M.users),
       'itens_treino': len(M.items), 'escolhido': [int(b.k), int(b.shrink)],
       'pop_teste': mp, 'knn_teste': mk, 'pop_validacao': mv, 'dif_ndcg_ic': ic,
       'grade': g[['k', 'shrink', 'ndcg10']].to_dict('records'), 'seg': time.perf_counter() - t0}
(R / 'reproducao_e2.json').write_text(json.dumps(out, indent=2, default=str), encoding='utf-8')
print(json.dumps(out, indent=1, default=str))
