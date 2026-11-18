"""Etapa 4 - consolidação do modelo: grades ampliadas, fusão, escolha do modelo final, curvas @K e robustez.

Uso:
  python etapa4.py grade <dados> <res_etapa3> <saida> <semente>   # grade ampliada na validação (uma semente)
  python etapa4.py final <dados> <res_etapa3> <saida>             # seleção, teste, robustez temporal e exemplos

Disciplina de avaliação: toda escolha (configuração de maior acerto, pesos da fusão e modelo final) usa apenas
a VALIDAÇÃO do protocolo A. O teste é lido uma única vez, depois de fixadas as escolhas.

Regra do modelo final (fixada antes da leitura do teste):
  c* = configuração de maior NDCG@10 médio de validação entre as cinco sementes;
  candidatos = configurações da fronteira de Pareto (NDCG@10 x cobertura, médias de validação) com cobertura
               maior que a de c*;
  F  = candidato de MAIOR COBERTURA cuja diferença de NDCG@10 para c*, na validação, é não inferior com
       margem DELTA: limite inferior do IC95% (bootstrap de usuários, média das cinco sementes) > -DELTA.
  DELTA = 0,005, a perda do ItemKNN em relação à popularidade que a etapa 2 tratou como déficit relevante.
"""
import hashlib, json, platform, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix, save_npz

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comum as c

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding='utf-8')
    except Exception:
        pass
FASE = sys.argv[1]
D, R3, OUT = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
OUT.mkdir(parents=True, exist_ok=True)
import os
SEMENTES = [42] if os.environ.get('E4_SMOKE') else [42, 7, 13, 2024, 2026]
DELTA = 0.005
KMAX = 20
REFS = ('Popularidade', 'Aleatório')
EXT = {
    'RP3β': [dict(alpha=0.1, beta=b, k=k) for b in (0.0, 0.1, 0.2, 0.3, 0.4) for k in (50, 200)]
            + [dict(alpha=a, beta=b, k=k) for a in (0.25, 0.5, 0.75, 1.0) for b in (-0.2, -0.1) for k in (50, 200)]
            + [dict(alpha=a, beta=b, k=k) for a in (0.25, 0.5) for b in (0.15, 0.25) for k in (50, 200)]
            + [dict(alpha=a, beta=b, k=500) for a in (0.25, 0.5) for b in (0.0, 0.1, 0.2, 0.3)],
    'EASE': [dict(lam=l) for l in (10000, 20000, 50000)],
    'iALS': [dict(fatores=64, reg=10000, alpha=a, iters=10) for a in (20, 100)]
            + [dict(fatores=64, reg=r, alpha=5, iters=10) for r in (1000, 3000)],
    'ItemKNN': [dict(k=k, shrink=s) for k in (20, 50) for s in (2000, 5000)],
    'ItemKNN+Pop': [dict(k=100, shrink=50, beta=b) for b in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)],
}
PESOS = [(a / 4, b / 4, (4 - a - b) / 4) for a in range(5) for b in range(5 - a)]  # RP3β, EASE, iALS


def construir(fam, M, h, C=None):
    if fam == 'Popularidade':
        return c.Popularidade(M)
    if fam == 'ItemKNN':
        return c.ItemKNN(M, h['k'], h['shrink'], C)
    if fam == 'ItemKNN+Pop':
        return c.Hibrido(c.ItemKNN(M, h['k'], h['shrink'], C), h['beta'])
    if fam == 'RP3β':
        return c.RP3beta(M, h['alpha'], h['beta'], h['k'])
    if fam == 'EASE':
        return c.EASE(M, h['lam'], C)
    if fam == 'iALS':
        return c.IALS(M, h['fatores'], h['reg'], h['alpha'], h['iters'])
    raise ValueError(fam)


def znorm(S):
    """Padroniza os escores de cada usuário sobre o catálogo (média 0, desvio 1)."""
    S = np.asarray(S, dtype=float)
    mu = S.mean(axis=1, keepdims=True); sd = S.std(axis=1, keepdims=True); sd[sd == 0] = 1.0
    return (S - mu) / sd


class Fusao:
    """Fusão de escores: soma ponderada dos escores padronizados por usuário de vários modelos."""
    nome = 'Fusão'

    def __init__(self, modelos, pesos, cache=None, pos=None):
        self.modelos = modelos; self.pesos = pesos; self.M = modelos[0].M if modelos else None
        self.cache = cache; self.pos = pos

    def escores(self, us):
        if self.cache is not None:
            return sum(w * self.cache[k][self.pos[us]].astype(float) for k, w in enumerate(self.pesos) if w)
        return sum(w * znorm(m.escores(us)) for m, w in zip(self.modelos, self.pesos) if w)


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


def avaliar_k(modelo, M, parte, bloqueio=None, lote=1024):
    """Protocolo A (um alvo por usuário): posição do alvo (1..KMAX; 0 = fora) e lista top-10 de cada usuário.

    Mesma ordenação de comum.avaliar: escore, popularidade no treino, identificador crescente.
    """
    alvo = parte.set_index('user').item
    usuarios = [u for u in alvo.index if u in M.um]
    us = np.array([M.um[u] for u in usuarios]); idx = np.arange(len(M.items))
    pos = np.zeros(len(us), dtype=int); listas = np.zeros((len(us), c.TOP), dtype=int)
    hist = np.zeros(len(us), dtype=int); auc = np.full(len(us), np.nan)
    for ini in range(0, len(us), lote):
        bloco = us[ini:ini + lote]
        S = modelo.escores(bloco).astype(float)
        for n, u in enumerate(bloco):
            a = S[n]; conhecidos = M.X[u].indices
            a[conhecidos] = -np.inf
            rej = bloqueio.get(usuarios[ini + n], ()) if bloqueio is not None else ()
            if len(rej):
                a[list(rej)] = -np.inf
            ordem = np.lexsort((idx, -M.pop, -a)); rank = ordem[:KMAX]
            t = M.im.get(alvo.iloc[ini + n], -1)
            w = np.flatnonzero(rank == t)
            pos[ini + n] = w[0] + 1 if len(w) else 0
            if t >= 0:  # AUC: fração dos demais candidatos ordenados abaixo do alvo
                ncand = len(idx) - len(conhecidos) - len(rej)
                auc[ini + n] = 1 - np.flatnonzero(ordem == t)[0] / (ncand - 1)
            listas[ini + n] = rank[:c.TOP]; hist[ini + n] = len(conhecidos)
    return pd.DataFrame({'user': usuarios, 'pos': pos, 'hist': hist, 'auc': auc}), listas


def metricas_pos(pos, k=10):
    """Métricas @k a partir da posição do único alvo (0 = ausente)."""
    dentro = (pos > 0) & (pos <= k)
    ndcg = np.where(dentro, 1 / np.log2(np.maximum(pos, 1) + 1), 0.0)
    rr = np.where(dentro, 1 / np.maximum(pos, 1), 0.0)
    hit = dentro.astype(float)
    prec = hit / k; rec = hit
    f1 = np.where(dentro, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    return ndcg, hit, rr, prec, rec, f1


def p(*a):
    print(*a, flush=True)


# =====================================================================================================
df, aud = c.carregar(D / 'interacoes.csv.gz')
pos_df = df[df.positive].copy()

if FASE == 'grade':
    s = int(sys.argv[5]); t0 = time.perf_counter()
    P = c.particao_aleatoria(pos_df, s); M = c.matriz(P.treino); C = (M.X.T @ M.X).toarray()
    _, _, lp = c.avaliar(c.Popularidade(M), M, P.validacao, guardar_listas=True)
    lp = np.array(lp); linhas = []
    for fam, grade in EXT.items():
        for h in grade:
            ti = time.perf_counter()
            m, _, ls = c.avaliar(construir(fam, M, h, C), M, P.validacao, guardar_listas=True)
            sob = float(np.mean([len(set(a) & set(b)) / c.TOP for a, b in zip(np.array(ls), lp)]))
            linhas.append({'familia': fam, 'hp': json.dumps(h), **m, 'sobreposicao_pop': sob,
                           'seg': time.perf_counter() - ti, 'semente': s, 'origem': 'etapa 4'})
        p(f'[s{s}] {fam}: {time.perf_counter() - t0:.0f}s')
    pd.DataFrame(linhas).to_csv(OUT / f'ext_grade_s{s}.csv', index=False)
    sys.exit(0)

# ============================================== FASE FINAL ==============================================
t_inicio = time.perf_counter()
G3 = pd.read_csv(R3 / 'A_validacao_grades.csv'); G3['origem'] = 'etapa 3'
G3 = G3[G3.familia != 'ItemKNN+Pop']  # na etapa 3 a base do híbrido variava com a semente; a etapa 4 reavaliou com base fixa
G4 = pd.concat([pd.read_csv(OUT / f'ext_grade_s{s}.csv') for s in SEMENTES])
G = pd.concat([G3, G4], ignore_index=True)
G = G[G.semente.isin(SEMENTES)]


def agrega(G):
    a = G.groupby(['familia', 'hp']).agg(n=('semente', 'size'), ndcg=('ndcg10', 'mean'), ndcg_sd=('ndcg10', 'std'),
                                         cob=('cobertura', 'mean'), gini=('gini_exposicao', 'mean'),
                                         nov=('novidade', 'mean'), cab=('participacao_cabeca', 'mean'),
                                         origem=('origem', 'first')).reset_index()
    return a[a.n == len(SEMENTES)].reset_index(drop=True)


AG = agrega(G)
comp = {}
for fam in ('RP3β', 'EASE', 'iALS'):
    r = AG[AG.familia == fam].sort_values('ndcg', ascending=False, kind='stable').iloc[0]
    comp[fam] = json.loads(r.hp)
p('componentes da fusão', comp)

# ---------------- Passo 1: fusão avaliada na validação (pesos escolhidos pela média das sementes)
ctx = {}; lin_f = []; per_val = {}
for s in SEMENTES:
    P = c.particao_aleatoria(pos_df, s); M = c.matriz(P.treino); C = (M.X.T @ M.X).toarray()
    us_el = np.array([M.um[u] for u in P.validacao.user])
    posmap = -np.ones(M.X.shape[0], dtype=int); posmap[us_el] = np.arange(len(us_el))
    Z = []
    for fam in ('RP3β', 'EASE', 'iALS'):
        mod = construir(fam, M, comp[fam], C)
        Z.append(np.vstack([znorm(mod.escores(us_el[i:i + 512])) for i in range(0, len(us_el), 512)]).astype(np.float32))
    ctx[s] = dict(P=P, M=M, C=C, Z=Z, posmap=posmap)
    for w in PESOS:
        if max(w) == 1.0:
            continue  # modelos puros já estão na grade
        m, per = c.avaliar(Fusao([], w, cache=Z, pos=posmap), M, P.validacao)
        hp = json.dumps({'RP3β': w[0], 'EASE': w[1], 'iALS': w[2]})
        lin_f.append({'familia': 'Fusão', 'hp': hp, **m, 'semente': s, 'origem': 'etapa 4'})
        per_val[('Fusão', hp, s)] = per.set_index('user').ndcg10
    p(f'[s{s}] fusão na validação: {time.perf_counter() - t_inicio:.0f}s')
G = pd.concat([G, pd.DataFrame(lin_f)], ignore_index=True)
G.to_csv(OUT / 'validacao_grades_completa.csv', index=False)
AG = agrega(G)
AG.to_csv(OUT / 'validacao_media_sementes.csv', index=False)

# ---------------- H4: a ampliação das grades na direção da fronteira traz ganho ou reproduz a popularidade?
def fora_da_grade_e3(fam, h):
    if fam == 'RP3β':
        return h['alpha'] < 0.25 or h['beta'] < 0 or h['beta'] > 0.4 or h['k'] > 200
    if fam == 'EASE':
        return h['lam'] > 5000
    if fam == 'iALS':
        return h['reg'] > 3000 or h['alpha'] < 20
    if fam == 'ItemKNN':
        return h['shrink'] > 1000
    if fam == 'ItemKNN+Pop':
        return h['beta'] > 5
    return False


pop_listas = {s: np.array(c.avaliar(c.Popularidade(ctx[s]['M']), ctx[s]['M'], ctx[s]['P'].validacao,
                                    guardar_listas=True)[2]) for s in SEMENTES}


def sobreposicao(fam, hp_txt):
    v = []
    for s in SEMENTES:
        _, _, ls = c.avaliar(construir(fam, ctx[s]['M'], json.loads(hp_txt), ctx[s]['C']), ctx[s]['M'],
                             ctx[s]['P'].validacao, guardar_listas=True)
        v.append(np.mean([len(set(a) & set(b)) / c.TOP for a, b in zip(np.array(ls), pop_listas[s])]))
    return float(np.mean(v))


h4 = []
for fam in ('RP3β', 'EASE', 'iALS', 'ItemKNN', 'ItemKNN+Pop'):
    a = AG[AG.familia == fam].copy()
    a['fora'] = [fora_da_grade_e3(fam, json.loads(x)) for x in a.hp]
    b0 = a[~a.fora].sort_values('ndcg', ascending=False, kind='stable').iloc[0]
    b1 = a.sort_values('ndcg', ascending=False, kind='stable').iloc[0]
    s0 = sobreposicao(fam, b0.hp); s1 = s0 if b1.hp == b0.hp else sobreposicao(fam, b1.hp)
    h4.append({'familia': fam, 'hp_grade_e3': b0.hp, 'ndcg_grade_e3': b0.ndcg, 'cob_grade_e3': b0.cob,
               'sobrep_grade_e3': s0, 'hp_ampliada': b1.hp, 'ndcg_ampliada': b1.ndcg, 'cob_ampliada': b1.cob,
               'sobrep_ampliada': s1, 'ganho': b1.ndcg - b0.ndcg, 'otimo_fora_da_grade_e3': bool(b1.fora),
               'configs_fora': int(a.fora.sum())})
    p('H4', fam, round(b1.ndcg - b0.ndcg, 5), bool(b1.fora))
pd.DataFrame(h4).to_csv(OUT / 'h4_ampliacao.csv', index=False)

# ---------------- Passo 2: c*, fronteira de Pareto e candidatos
EL = AG[~AG.familia.isin(REFS)].copy()
cstar = EL.sort_values('ndcg', ascending=False, kind='stable').iloc[0]
pop_val = float(AG[AG.familia == 'Popularidade'].ndcg.iloc[0])
front = []; melhor = -np.inf
for _, r in EL.sort_values(['cob', 'ndcg'], ascending=[False, False], kind='stable').iterrows():
    if r.ndcg > melhor:
        front.append(r); melhor = r.ndcg
FRONT = pd.DataFrame(front)
CAND = FRONT[(FRONT.cob > cstar.cob) & (FRONT.ndcg >= pop_val)]
p('c*', cstar.familia, cstar.hp, round(cstar.ndcg, 5), round(cstar.cob, 4), '| candidatos:', len(CAND))


def modelo_cfg(fam, hp_txt, s):
    h = json.loads(hp_txt); x = ctx[s]
    if fam == 'Fusão':
        return Fusao([], (h['RP3β'], h['EASE'], h['iALS']), cache=x['Z'], pos=x['posmap'])
    return construir(fam, x['M'], h, x['C'])


def val_por_usuario(fam, hp_txt, s):
    k = (fam, hp_txt, s)
    if k not in per_val:
        _, per = c.avaliar(modelo_cfg(fam, hp_txt, s), ctx[s]['M'], ctx[s]['P'].validacao)
        per_val[k] = per.set_index('user').ndcg10
    return per_val[k]


usuarios_el = sorted(ctx[42]['P'].validacao.user)
base = np.mean([val_por_usuario(cstar.familia, cstar.hp, s).loc[usuarios_el].values for s in SEMENTES], axis=0)
sel = []
for _, r in CAND.iterrows():
    v = np.mean([val_por_usuario(r.familia, r.hp, s).loc[usuarios_el].values for s in SEMENTES], axis=0)
    d, lo, hi, pv = c.bootstrap_pareado(v, base, semente=42)
    sel.append({'familia': r.familia, 'hp': r.hp, 'val_ndcg': r.ndcg, 'val_cob': r.cob, 'dif_cstar': d,
                'ic_inf': lo, 'ic_sup': hi, 'nao_inferior': bool(lo > -DELTA)})
SEL = pd.DataFrame(sel)
SEL.to_csv(OUT / 'selecao_modelo_final.csv', index=False)
adm = SEL[SEL.nao_inferior] if len(SEL) else SEL
if len(adm):
    f = adm.sort_values('val_cob', ascending=False, kind='stable').iloc[0]
    FINAL = (f.familia, f.hp)
else:
    FINAL = (cstar.familia, cstar.hp)
CSTAR = (cstar.familia, cstar.hp)
fus = AG[AG.familia == 'Fusão'].sort_values('ndcg', ascending=False, kind='stable').iloc[0]
FUSAO = ('Fusão', fus.hp)
p('FINAL', FINAL, '| c*', CSTAR, '| fusão', FUSAO)

# ---------------- Passo 3: teste (lido uma única vez), com bloqueio de rejeitados
MODELOS = {'Popularidade': ('Popularidade', '{}'), 'ItemKNN (etapa 2)': ('ItemKNN', json.dumps({'k': 100, 'shrink': 50})),
           'Máximo acerto (c*)': CSTAR, 'Modelo final (F)': FINAL, 'Fusão': FUSAO}
teste = []; pos_usuario = {}; listas_s42 = {}
for s in SEMENTES:
    x = ctx[s]; M, P = x['M'], x['P']; rej = rejeitados(df, M)
    res = {}
    for nome, (fam, hp) in MODELOS.items():
        per, ls = avaliar_k(modelo_cfg(fam, hp, s), M, P.teste, bloqueio=rej)
        res[nome] = (per, ls)
        if s == 42:
            listas_s42[nome] = (per.user.tolist(), ls)
    lp = res['Popularidade'][1]
    b_nd = metricas_pos(res['Popularidade'][0].pos.values)[0]
    pv = {}; ics = {}
    for nome, (per, ls) in res.items():
        nd, hit, rr, pr, rc, f1 = metricas_pos(per.pos.values)
        ex = c.exposicao(ls.ravel(), M)
        linha = {'modelo': nome, 'familia': MODELOS[nome][0], 'hp': MODELOS[nome][1], 'semente': s,
                 'ndcg10': nd.mean(), 'hit10': hit.mean(), 'mrr10': rr.mean(), 'precision10': pr.mean(),
                 'recall10': rc.mean(), 'f1_10': f1.mean(), 'auc': float(np.nanmean(per.auc.values)),
                 'auc_usuarios': int(per.auc.notna().sum()), **ex,
                 'sobreposicao_pop': float(np.mean([len(set(a) & set(b)) / c.TOP for a, b in zip(ls, lp)])),
                 'usuarios': len(per)}
        for k in range(1, KMAX + 1):
            ndk, hk = metricas_pos(per.pos.values, k)[:2]
            linha[f'ndcg@{k}'] = ndk.mean(); linha[f'hit@{k}'] = hk.mean()
        teste.append(linha)
        pos_usuario[(nome, s)] = per.set_index('user')
        if nome != 'Popularidade':
            d, lo, hi, pval = c.bootstrap_pareado(nd, b_nd, semente=s)
            ics[nome] = (d, lo, hi); pv[nome] = pval
    adj = c.holm(pv)
    for l in teste:
        if l['semente'] == s and l['modelo'] in ics:
            l['dif_pop'], l['ic_inf'], l['ic_sup'] = ics[l['modelo']]; l['p_holm'] = adj[l['modelo']]
    p(f'[s{s}] teste concluído')
T = pd.DataFrame(teste); T.to_csv(OUT / 'teste_modelos_finais.csv', index=False)


def ndcg_medio_usuario(nome):
    return np.mean([metricas_pos(pos_usuario[(nome, s)].loc[usuarios_el].pos.values)[0] for s in SEMENTES], axis=0)


agreg = {}
u_pop = ndcg_medio_usuario('Popularidade'); u_c = ndcg_medio_usuario('Máximo acerto (c*)')
for nome in MODELOS:
    if nome == 'Popularidade':
        continue
    u = ndcg_medio_usuario(nome)
    d, lo, hi, pv_ = c.bootstrap_pareado(u, u_pop, semente=42)
    agreg[nome] = {'dif_pop': d, 'ic_inf': lo, 'ic_sup': hi, 'p': pv_}
    if nome != 'Máximo acerto (c*)':
        d2, lo2, hi2, pv2 = c.bootstrap_pareado(u, u_c, semente=42)
        agreg[nome].update(dif_cstar=d2, ic_inf_cstar=lo2, ic_sup_cstar=hi2, p_cstar=pv2)
hist = pos_usuario[('Popularidade', 42)].loc[usuarios_el]['hist'].values
faixa = pd.cut(hist, [2, 3, 5, 100], labels=['3', '4-5', '6 ou mais'])
seg = []
for nome in MODELOS:
    u = ndcg_medio_usuario(nome)
    for fx in ['3', '4-5', '6 ou mais']:
        m = np.asarray(faixa == fx)
        linha = {'modelo': nome, 'faixa': fx, 'usuarios': int(m.sum()), 'ndcg10': float(u[m].mean())}
        if nome != 'Popularidade':
            d, lo, hi, _ = c.bootstrap_pareado(u[m], u_pop[m], semente=42)
            linha.update(dif_pop=d, ic_inf=lo, ic_sup=hi)
        seg.append(linha)
pd.DataFrame(seg).to_csv(OUT / 'segmentos_5_sementes.csv', index=False)
pd.DataFrame({'user': usuarios_el, 'hist': hist, **{n: ndcg_medio_usuario(n) for n in MODELOS}}).to_csv(
    OUT / 'por_usuario_media_5_sementes.csv', index=False)

# ---------------- Passo 4: robustez temporal dos modelos escolhidos (configuração fixa)
temporal = []
for rot, basepos in [('B-com-inferencia', pos_df), ('B-somente-datados', pos_df[~pos_df.ano_inferido])]:
    P, cortes = c.particao_temporal(basepos)
    tv = pd.concat([P.treino, P.validacao]); M = c.matriz(tv); C = (M.X.T @ M.X).toarray()
    rej = rejeitados(df, M, P.teste.data.min())
    per = {}; feitos = {}
    for nome, (fam, hp) in MODELOS.items():
        h = json.loads(hp)
        if (fam, hp) not in feitos:
            if fam == 'Fusão':
                mod = Fusao([construir(k, M, comp[k], C) for k in ('RP3β', 'EASE', 'iALS') if h[k]],
                            [h[k] for k in ('RP3β', 'EASE', 'iALS') if h[k]])
            else:
                mod = construir(fam, M, h, C)
            feitos[(fam, hp)] = c.avaliar(mod, M, P.teste, bloqueio=rej)
        m, pu = feitos[(fam, hp)]
        per[nome] = pu.set_index('user').ndcg10
        temporal.append({'protocolo': rot, 'modelo': nome, **m})
    pv = {}
    for nome in MODELOS:
        if nome == 'Popularidade':
            continue
        d, lo, hi, pval = c.bootstrap_pareado(per[nome].loc[per['Popularidade'].index].values,
                                              per['Popularidade'].values, semente=42)
        pv[nome] = pval
        for l in temporal:
            if l['protocolo'] == rot and l['modelo'] == nome:
                l.update(dif_pop=d, ic_inf=lo, ic_sup=hi)
    adj = c.holm(pv)
    for l in temporal:
        if l['protocolo'] == rot and l['modelo'] in adj:
            l['p_holm'] = adj[l['modelo']]
    p(rot, 'concluído')
pd.DataFrame(temporal).to_csv(OUT / 'temporal_modelos_finais.csv', index=False)

# ---------------- Passo 5: consolidação do modelo final com todos os positivos
M = c.matriz(pos_df); C = (M.X.T @ M.X).toarray()
fam, hp = FINAL; h = json.loads(hp)
t0 = time.perf_counter()
if fam == 'Fusão':
    mod = Fusao([construir(k, M, comp[k], C) for k in ('RP3β', 'EASE', 'iALS') if h[k]],
                [h[k] for k in ('RP3β', 'EASE', 'iALS') if h[k]])
else:
    mod = construir(fam, M, h, C)
t_fit = time.perf_counter() - t0
rng = np.random.default_rng(0); us = rng.choice(M.X.shape[0], 1000, replace=False)
t0 = time.perf_counter(); S = mod.escores(us)
for n, u in enumerate(us):
    a = S[n].astype(float); a[M.X[u].indices] = -np.inf
    np.lexsort((np.arange(len(M.items)), -M.pop, -a))[:10]
lat = (time.perf_counter() - t0) / len(us) * 1000
MF = OUT / 'modelo_final'; MF.mkdir(exist_ok=True)
save_npz(MF / 'matriz_positivos.npz', M.X)
W = getattr(mod, 'S', None)
if W is None and hasattr(mod, 'B'):
    W = csr_matrix(mod.B)
if fam == 'ItemKNN+Pop':
    W = mod.base.S
if W is not None:
    save_npz(MF / 'pesos_item_item.npz', csr_matrix(W))
rej_all = {}
for u, i in zip(df[~df.positive].user, df[~df.positive].item):
    if i in M.im:
        rej_all.setdefault(u, []).append(M.im[i])
(MF / 'mapas.json').write_text(json.dumps({'users': M.users, 'items': M.items, 'rejeitados': rej_all}),
                               encoding='utf-8')
meta = {'familia': fam, 'hiperparametros': h, 'regra': 'maior cobertura entre as configurações não inferiores a c* '
        f'na validação (margem {DELTA})', 'usuarios': len(M.users), 'jogos': len(M.items), 'interacoes': int(M.X.nnz),
        'segundos_ajuste': t_fit, 'ms_por_usuario_top10': lat, 'nnz_pesos': int(W.nnz) if W is not None else None}
(MF / 'modelo.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')

# ---------------- Passo 6: exemplos de recomendação (semente 42; regra fixa, sem escolha manual)
tit = pd.read_csv(D / 'titulos.csv', dtype=str).drop_duplicates('item').set_index('item').title.to_dict()
x = ctx[42]; M42, P42 = x['M'], x['P']


def nome_jogo(j):
    return tit.get(M42.items[j], f'App {M42.items[j]}')


us42, lsF = listas_s42['Modelo final (F)']; _, lsP = listas_s42['Popularidade']
alvos42 = P42.teste.set_index('user')['item']
perF = pos_usuario[('Modelo final (F)', 42)]; perP = pos_usuario[('Popularidade', 42)]
exemplos = []
for fx, (lo_, hi_) in {'3': (3, 3), '4-5': (4, 5), '6 ou mais': (6, 99)}.items():
    # regra: primeiro usuário da faixa, em ordem de pseudônimo, cujo alvo e histórico têm nome nos metadados
    for n, u in enumerate(us42):
        hh = perF.loc[u, 'hist']
        alvo = alvos42.loc[u]
        if lo_ <= hh <= hi_ and alvo in tit and all(M42.items[j] in tit for j in M42.X[M42.um[u]].indices):
            exemplos.append({'faixa': fx, 'usuario': u[:8], 'historico': [nome_jogo(j) for j in M42.X[M42.um[u]].indices],
                             'alvo': tit[alvo], 'pos_final': int(perF.loc[u, 'pos']), 'pos_pop': int(perP.loc[u, 'pos']),
                             'final': [nome_jogo(j) for j in lsF[n]], 'popularidade': [nome_jogo(j) for j in lsP[n]]})
            break
(OUT / 'exemplos_recomendacao.json').write_text(json.dumps(exemplos, indent=1, ensure_ascii=False), encoding='utf-8')

resumo = {'delta': DELTA, 'componentes_fusao': comp, 'cstar': {'familia': CSTAR[0], 'hp': json.loads(CSTAR[1]),
          'val_ndcg': float(cstar.ndcg), 'val_cob': float(cstar.cob)},
          'final': {'familia': FINAL[0], 'hp': json.loads(FINAL[1])}, 'fusao': {'hp': json.loads(FUSAO[1]),
          'val_ndcg': float(fus.ndcg), 'val_cob': float(fus.cob)}, 'pop_val_ndcg': pop_val,
          'candidatos': int(len(CAND)), 'fronteira': int(len(FRONT)), 'configs_elegiveis': int(len(EL)),
          'configs_etapa4': int((AG.origem == 'etapa 4').sum()), 'agregado_usuarios': agreg, 'modelo': meta,
          'versoes': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                      'scipy': scipy.__version__},
          'hash_dados': hashlib.sha256((D / 'interacoes.csv.gz').read_bytes()).hexdigest(),
          'tempo_final_s': time.perf_counter() - t_inicio}
FRONT.to_csv(OUT / 'fronteira_pareto.csv', index=False)
(OUT / 'resumo_etapa4.json').write_text(json.dumps(resumo, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
p(json.dumps({k: resumo[k] for k in ('cstar', 'final', 'fusao', 'agregado_usuarios')}, indent=1, ensure_ascii=False))
