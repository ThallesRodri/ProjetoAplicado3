"""Etapa 3 - ajuste do pipeline, reavaliação e testes de robustez.

Uso: python etapa3.py <pasta_dados> <pasta_resultados>

Protocolo A (comparável à etapa 2): para cada semente, um positivo de validação e
um de teste por usuário com >= 5 positivos. Hiperparâmetros escolhidos apenas na
validação; o teste é avaliado uma única vez por configuração escolhida.
Protocolo B (robustez temporal): divisão global por data (80/10/10), com a
hipótese H-T explicitada para avaliações sem ano.
"""
import json, platform, sys, time, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comum as c

D = Path(sys.argv[1]); R = Path(sys.argv[2]); R.mkdir(parents=True, exist_ok=True)
SEMENTES = [42, 7, 13, 2024, 2026]
COB_MIN = 0.50  # ponto de operação B: não perder a cobertura obtida na etapa 2
GRADES = {
    'ItemKNN': [dict(k=k, shrink=s) for k in (20, 50, 100, 200, 500)
                for s in (0, 10, 50, 100, 200, 500, 1000)],
    'RP3β': [dict(alpha=a, beta=b, k=k) for a in (0.25, 0.5, 0.75, 1.0)
             for b in (0.0, 0.1, 0.2, 0.3, 0.4) for k in (50, 200)],
    'EASE': [dict(lam=l) for l in (10, 50, 100, 250, 500, 1000, 2500, 5000)],
    'iALS': [dict(fatores=64, reg=r, alpha=a, iters=10) for r in (100, 300, 1000, 3000)
             for a in (20, 100)],
}
BETAS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
log = open(R / 'execucao.log', 'w', encoding='utf-8')


def p(*a):
    print(*a, flush=True); print(*a, file=log, flush=True)


def construir(fam, M, h, C):
    if fam == 'ItemKNN':
        return c.ItemKNN(M, h['k'], h['shrink'], C)
    if fam == 'RP3β':
        return c.RP3beta(M, h['alpha'], h['beta'], h['k'])
    if fam == 'EASE':
        return c.EASE(M, h['lam'], C)
    if fam == 'iALS':
        return c.IALS(M, h['fatores'], h['reg'], h['alpha'], h['iters'])
    raise ValueError(fam)


def selecionar(grade_df):
    """A: maior NDCG@10 de validação. B: maior NDCG@10 com cobertura >= COB_MIN."""
    g = grade_df.sort_values('ndcg10', ascending=False, kind='stable')
    a = g.iloc[0]
    gb = g[g.cobertura >= COB_MIN]
    b = gb.iloc[0] if len(gb) else None
    return a, b


def rejeitados(df, M, ate=None):
    neg = df[~df.positive]
    if ate is not None:
        neg = neg[neg.data < ate]
    out = {}
    for u, i in zip(neg.user, neg.item):
        j = M.im.get(i)
        if j is not None:
            out.setdefault(u, set()).add(j)
    return out


def rodada(df, P, rotulo, retreinar=False, semente_boot=42):
    """Executa grade em validação, seleciona A/B e avalia no teste."""
    t0 = time.perf_counter()
    M = c.matriz(P.treino); C = (M.X.T @ M.X).toarray()
    c.definir_rejeitados(rejeitados(df, M, P.validacao.data.min() if retreinar else None))
    linhas = []; modelos_val = {}
    for fam, grade in GRADES.items():
        for h in grade:
            ti = time.perf_counter()
            mod = construir(fam, M, h, C)
            m, _ = c.avaliar(mod, M, P.validacao)
            linhas.append({'familia': fam, 'hp': json.dumps(h), **m,
                           'seg': time.perf_counter() - ti})
        p(f'  [{rotulo}] {fam}: melhor val NDCG='
          f'{max(l["ndcg10"] for l in linhas if l["familia"] == fam):.5f}')
    grade_df = pd.DataFrame(linhas)
    # Híbrido: parte do melhor ItemKNN (critério A) e ajusta beta na validação.
    knn_a, _ = selecionar(grade_df[grade_df.familia == 'ItemKNN'])
    hk = json.loads(knn_a.hp); base = construir('ItemKNN', M, hk, C)
    for b in BETAS:
        m, _ = c.avaliar(c.Hibrido(base, b), M, P.validacao)
        linhas.append({'familia': 'ItemKNN+Pop', 'hp': json.dumps({**hk, 'beta': b}), **m, 'seg': 0})
    for nome, mod in [('Popularidade', c.Popularidade(M)), ('Aleatório', c.Aleatorio(M, 0))]:
        m, _ = c.avaliar(mod, M, P.validacao)
        linhas.append({'familia': nome, 'hp': '{}', **m, 'seg': 0})
    grade_df = pd.DataFrame(linhas)
    # Seleção
    escolhas = []
    for fam in ['Popularidade', 'Aleatório', 'ItemKNN', 'ItemKNN+Pop', 'RP3β', 'EASE', 'iALS']:
        a, b = selecionar(grade_df[grade_df.familia == fam])
        escolhas.append((fam, 'A', a))
        if b is not None and fam not in ('Popularidade', 'Aleatório') and b.hp != a.hp:
            escolhas.append((fam, 'B', b))
        elif b is not None and fam not in ('Popularidade', 'Aleatório'):
            escolhas.append((fam, 'B', a))
    # Teste: se retreinar, a matriz passa a ser treino+validação (protocolo temporal).
    if retreinar:
        tv = pd.concat([P.treino, P.validacao])
        M = c.matriz(tv); C = (M.X.T @ M.X).toarray()
        c.definir_rejeitados(rejeitados(df, M, P.teste.data.min()))
    res = []; per_usuario = {}
    for fam, ponto, row in escolhas:
        h = json.loads(row.hp)
        if fam == 'Popularidade':
            mod = c.Popularidade(M)
        elif fam == 'Aleatório':
            mod = c.Aleatorio(M, 0)
        elif fam == 'ItemKNN+Pop':
            mod = c.Hibrido(construir('ItemKNN', M, {k: h[k] for k in ('k', 'shrink')}, C), h['beta'])
        else:
            mod = construir(fam, M, h, C)
        ti = time.perf_counter()
        m, per = c.avaliar(mod, M, P.teste)
        lat = (time.perf_counter() - ti) / max(m['usuarios'], 1) * 1000
        res.append({'familia': fam, 'ponto': ponto, 'hp': row.hp, 'val_ndcg10': row.ndcg10,
                    'val_cobertura': row.cobertura, **m, 'ms_por_usuario': lat})
        per_usuario[(fam, ponto)] = per
        # Bloqueio de rejeitados explícitos (efeito isolado) nos pontos escolhidos.
        if fam not in ('Aleatório',):
            mb, perb = c.avaliar(mod, M, P.teste, bloqueio=c._rejeitados)
            res.append({'familia': fam, 'ponto': ponto + '+bloqueio', 'hp': row.hp,
                        'val_ndcg10': row.ndcg10, 'val_cobertura': row.cobertura, **mb,
                        'ms_por_usuario': np.nan})
            per_usuario[(fam, ponto + '+bloqueio')] = perb
    res = pd.DataFrame(res)
    # Inferência pareada contra popularidade (mesmos usuários), Holm por rodada.
    base = per_usuario[('Popularidade', 'A')].set_index('user').ndcg10
    pv = {}; ics = {}
    for (fam, ponto), per in per_usuario.items():
        if fam in ('Popularidade',) or 'bloqueio' in ponto:
            continue
        s = per.set_index('user').ndcg10.loc[base.index]
        d, lo, hi, pval = c.bootstrap_pareado(s.values, base.values, semente=semente_boot)
        ics[(fam, ponto)] = (d, lo, hi); pv[f'{fam}|{ponto}'] = pval
    adj = c.holm(pv)
    res['dif_ndcg_vs_pop'] = [ics.get((f, pt), (np.nan,) * 3)[0] for f, pt in zip(res.familia, res.ponto)]
    res['ic95_inf'] = [ics.get((f, pt), (np.nan,) * 3)[1] for f, pt in zip(res.familia, res.ponto)]
    res['ic95_sup'] = [ics.get((f, pt), (np.nan,) * 3)[2] for f, pt in zip(res.familia, res.ponto)]
    res['p_holm'] = [adj.get(f'{f}|{pt}', np.nan) for f, pt in zip(res.familia, res.ponto)]
    p(f'  [{rotulo}] concluído em {time.perf_counter() - t0:.0f}s')
    return grade_df, res, per_usuario, M


t_inicio = time.perf_counter()
df, aud = c.carregar(D / 'interacoes.csv.gz')
pos = df[df.positive].copy()
p('auditoria', aud)

# ---------------- Protocolo A: aleatório por usuário, 5 sementes ----------------
grades, testes = [], []
for s in SEMENTES:
    P = c.particao_aleatoria(pos, s)
    g, r, per, M = rodada(df, P, f'A-s{s}', semente_boot=s)
    g['semente'] = s; r['semente'] = s
    grades.append(g); testes.append(r)
    if s == 42:
        segs = []
        for (fam, ponto), pu in per.items():
            if 'bloqueio' in ponto:
                continue
            pu = pu.copy()
            pu['faixa'] = pd.cut(pu['hist'], [2, 3, 5, 100], labels=['3', '4-5', '6 ou mais'])
            t = pu.groupby('faixa', observed=True).agg(usuarios=('user', 'size'),
                                                      ndcg10=('ndcg10', 'mean'),
                                                      hit10=('hit10', 'mean')).reset_index()
            t['familia'] = fam; t['ponto'] = ponto; segs.append(t)
            pu.to_csv(R / f'por_usuario_s42_{fam}_{ponto}.csv', index=False)
        pd.concat(segs).to_csv(R / 'segmentos_s42.csv', index=False)
pd.concat(grades).to_csv(R / 'A_validacao_grades.csv', index=False)
TA = pd.concat(testes); TA.to_csv(R / 'A_teste.csv', index=False)

# ---------------- Protocolo B: temporal global ----------------
temporais = {}
for rot, base in [('B-com-inferencia', pos), ('B-somente-datados', pos[~pos.ano_inferido])]:
    P, cortes = c.particao_temporal(base)
    info = {'cortes': cortes, 'treino': len(P.treino), 'validacao': len(P.validacao),
            'teste': len(P.teste),
            'itens_teste_ausentes_hist': int((~P.teste.item.isin(
                pd.concat([P.treino, P.validacao]).item)).sum())}
    g, r, per, M = rodada(df, P, rot, retreinar=True)
    g.to_csv(R / f'{rot}_validacao_grades.csv', index=False)
    r.to_csv(R / f'{rot}_teste.csv', index=False)
    temporais[rot] = info
    p(rot, info)

# ---------------- Resumo ----------------
def agrega(T):
    cols = ['ndcg10', 'hit10', 'mrr10', 'cobertura', 'participacao_cabeca', 'gini_exposicao',
            'novidade', 'dif_ndcg_vs_pop']
    a = T.groupby(['familia', 'ponto'])[cols].agg(['mean', 'std'])
    a.columns = [f'{x}_{y}' for x, y in a.columns]
    sig = T.assign(sig=(T.ic95_inf > 0) & (T.p_holm < 0.05)).groupby(['familia', 'ponto']).sig.sum()
    neg = T.assign(n=(T.ic95_sup < 0)).groupby(['familia', 'ponto']).n.sum()
    a['sementes_melhor_que_pop'] = sig; a['sementes_pior_que_pop'] = neg
    return a.reset_index()

agrega(TA).to_csv(R / 'A_resumo_sementes.csv', index=False)
resumo = {
    'auditoria': aud,
    'datas': {'sem_ano': int(df.ano_inferido.sum()),
              'meses_sem_ano': sorted(df[df.ano_inferido].data.dt.month.unique().tolist()),
              'max_data_inferida': str(df[df.ano_inferido].data.max().date()),
              'max_data_datada': str(df[~df.ano_inferido].data.max().date())},
    'sementes': SEMENTES, 'cob_min': COB_MIN, 'grades': GRADES, 'betas': BETAS,
    'temporal': temporais,
    'versoes': {'python': platform.python_version(), 'numpy': np.__version__,
                'pandas': pd.__version__, 'scipy': scipy.__version__},
    'hash_dados': hashlib.sha256((D / 'interacoes.csv.gz').read_bytes()).hexdigest(),
    'tempo_total_s': time.perf_counter() - t_inicio}
(R / 'resumo_etapa3.json').write_text(json.dumps(resumo, indent=2, ensure_ascii=False, default=str),
                                      encoding='utf-8')
p('fim', resumo['tempo_total_s'])
