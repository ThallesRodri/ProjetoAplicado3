"""Funções compartilhadas da etapa 3 (Projeto Aplicado III - recomendação Steam).

Carregamento auditado, partições, modelos e métricas. Todas as estatísticas
que alimentam os modelos são calculadas exclusivamente sobre a matriz de treino
recebida; nenhuma função consulta a partição de teste para ajustar parâmetros.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags

TOP = 10
MESES = {m: i for i, m in enumerate(
    ['January', 'February', 'March', 'April', 'May', 'June', 'July',
     'August', 'September', 'October', 'November', 'December'], start=1)}
ANO_COLETA_INFERIDO = 2016  # hipótese H-T: avaliações sem ano pertencem ao ano da coleta


# --------------------------------------------------------------------------
# Dados
# --------------------------------------------------------------------------
def carregar(caminho):
    """Lê interações, audita ausentes/duplicidades e devolve (df_limpo, auditoria)."""
    raw = pd.read_csv(caminho, dtype={'user': str, 'item': str, 'posted': str})
    raw['positive'] = raw['positive'].astype(str).map({'True': True, 'False': False})
    aud = {'linhas_brutas': len(raw), 'usuarios_brutos': raw.user.nunique(),
           'itens_brutos': raw.item.nunique(),
           'ausentes': {c: int(raw[c].isna().sum()) for c in ['user', 'item', 'positive']}}
    assert sum(aud['ausentes'].values()) == 0
    conflitos = int((raw.groupby(['user', 'item']).positive.nunique() > 1).sum())
    assert conflitos == 0
    df = raw.drop_duplicates(['user', 'item']).copy()
    aud.update(duplicatas=len(raw) - len(df), conflitos=conflitos, unicas=len(df),
               positivas=int(df.positive.sum()), negativas=int((~df.positive).sum()))
    df = df.reset_index(drop=True)
    df[['data', 'ano_inferido']] = datas(df.posted)
    return df, aud


def datas(posted: pd.Series) -> pd.DataFrame:
    """Converte 'Posted Month D, YYYY.' em data. Sem ano -> ano de coleta inferido (marcado)."""
    rx = posted.str.extract(r'Posted (\w+) (\d{1,2})(?:, (\d{4}))?\.')
    mes = rx[0].map(MESES)
    dia = rx[1].astype(int)
    sem_ano = rx[2].isna()
    ano = rx[2].fillna(ANO_COLETA_INFERIDO).astype(int)
    data = pd.to_datetime(dict(year=ano, month=mes, day=dia), errors='coerce')
    assert data.notna().all()
    return pd.DataFrame({'data': data, 'ano_inferido': sem_ano})


@dataclass
class Particao:
    treino: pd.DataFrame
    validacao: pd.DataFrame
    teste: pd.DataFrame


def particao_aleatoria(pos: pd.DataFrame, semente: int, minimo: int = 5) -> Particao:
    """Protocolo da etapa 2: um positivo de validação e um de teste por usuário com >= minimo.

    A ordenação e o gerador replicam exatamente o script da etapa 2 quando semente = 42.
    """
    pos = pos.sort_values(['user', 'item']).reset_index(drop=True)
    cont = pos.groupby('user').size()
    eleg = set(cont[cont >= minimo].index)
    rng = np.random.default_rng(semente)
    v, t = [], []
    for u, p in pos.groupby('user', sort=True):
        if u in eleg:
            a, b = rng.choice(p.index.to_numpy(), size=2, replace=False)
            v.append(a); t.append(b)
    val = pos.loc[v].sort_values('user'); te = pos.loc[t].sort_values('user')
    tr = pos.drop(index=v + t)
    _disjuntas(tr, val, te)
    return Particao(tr, val, te)


def particao_temporal(pos: pd.DataFrame, q_val=0.80, q_teste=0.90) -> tuple[Particao, dict]:
    """Divisão global por data: treino < t_val <= validação < t_teste <= teste."""
    t_val = pos.data.quantile(q_val); t_te = pos.data.quantile(q_teste)
    tr = pos[pos.data < t_val]; val = pos[(pos.data >= t_val) & (pos.data < t_te)]
    te = pos[pos.data >= t_te]
    _disjuntas(tr, val, te)
    return Particao(tr, val, te), {'t_val': str(t_val.date()), 't_teste': str(t_te.date())}


def _disjuntas(*partes):
    pares = [set(zip(p.user, p.item)) for p in partes]
    for i in range(len(pares)):
        for j in range(i + 1, len(pares)):
            assert pares[i].isdisjoint(pares[j])


@dataclass
class Matriz:
    X: csr_matrix
    users: list
    items: list
    um: dict
    im: dict
    pop: np.ndarray


def matriz(treino: pd.DataFrame) -> Matriz:
    users = sorted(treino.user.unique())
    items = sorted(treino.item.unique(), key=int)
    um = {x: i for i, x in enumerate(users)}; im = {x: i for i, x in enumerate(items)}
    X = csr_matrix((np.ones(len(treino)), ([um[x] for x in treino.user],
                                           [im[x] for x in treino.item])),
                   shape=(len(users), len(items)))
    X.sum_duplicates(); X.data[:] = 1.0
    return Matriz(X, users, items, um, im, np.asarray(X.sum(axis=0)).ravel())


# --------------------------------------------------------------------------
# Modelos: cada um devolve uma função escores(indices_usuarios) -> ndarray denso
# --------------------------------------------------------------------------
def _topk_linhas(S: np.ndarray, k: int) -> np.ndarray:
    """Mantém os k maiores por linha; empate resolvido pelo menor índice (estável)."""
    if k >= S.shape[1]:
        return S
    ordem = np.argsort(-S, axis=1, kind='stable')[:, k:]
    S = S.copy()
    np.put_along_axis(S, ordem, 0.0, axis=1)
    return S


class Popularidade:
    nome = 'Popularidade'

    def __init__(self, M: Matriz):
        self.M = M

    def escores(self, us):
        return np.tile(self.M.pop.astype(float), (len(us), 1))


class Aleatorio:
    nome = 'Aleatório'

    def __init__(self, M: Matriz, semente=0):
        self.M = M; self.rng = np.random.default_rng(semente)

    def escores(self, us):
        return self.rng.random((len(us), len(self.M.items)))


class ItemKNN:
    """Vizinhança item-item: cosseno com redução por suporte c/(c+lambda) e top-k por item."""
    nome = 'ItemKNN'

    def __init__(self, M: Matriz, k: int, shrink: float, C=None):
        self.M = M; self.k = k; self.shrink = shrink
        C = (M.X.T @ M.X).toarray() if C is None else C
        C = C.copy(); np.fill_diagonal(C, 0)
        cos = C / np.sqrt(M.pop[:, None] * M.pop[None, :])
        S = cos * C / (C + shrink) if shrink else cos
        S = np.nan_to_num(S)
        self.S = csr_matrix(_topk_linhas(S, k))

    def escores(self, us):
        return (self.M.X[us] @ self.S).toarray()


class RP3beta:
    """Passeio aleatório de 3 passos com penalização de popularidade (Paudel et al., 2017)."""
    nome = 'RP3β'

    def __init__(self, M: Matriz, alpha: float, beta: float, k: int):
        self.M = M
        X = M.X
        du = np.asarray(X.sum(axis=1)).ravel(); di = M.pop
        Pui = diags(1 / du) @ X; Piu = diags(1 / di) @ X.T.tocsr()
        Pui = Pui.power(alpha); Piu = Piu.power(alpha)
        W = (Piu @ Pui).toarray()
        W = W / np.power(di, beta)[None, :]
        np.fill_diagonal(W, 0)
        self.S = csr_matrix(_topk_linhas(W, k))

    def escores(self, us):
        return (self.M.X[us] @ self.S).toarray()


class EASE:
    """Autoencoder linear raso de forma fechada (Steck, 2019): B = I - P diag(1/diag P)."""
    nome = 'EASE'

    def __init__(self, M: Matriz, lam: float, G=None):
        self.M = M
        G = (M.X.T @ M.X).toarray() if G is None else G.copy()
        G[np.diag_indices_from(G)] += lam
        P = np.linalg.inv(G)
        B = -P / np.diag(P)[None, :]
        np.fill_diagonal(B, 0)
        self.B = B

    def escores(self, us):
        return np.asarray(self.M.X[us] @ self.B)


class IALS:
    """Fatoração implícita por mínimos quadrados alternados (Hu; Koren; Volinsky, 2008)."""
    nome = 'iALS'

    def __init__(self, M: Matriz, fatores: int, reg: float, alpha: float,
                 iters: int = 15, semente: int = 0):
        self.M = M
        rng = np.random.default_rng(semente)
        X = M.X
        U = rng.normal(0, 0.01, (X.shape[0], fatores))
        V = rng.normal(0, 0.01, (X.shape[1], fatores))
        XT = X.T.tocsr()
        for _ in range(iters):
            U = self._passo(X, V, reg, alpha)
            V = self._passo(XT, U, reg, alpha)
        self.U, self.V = U, V

    @staticmethod
    def _passo(R, Y, reg, alpha, celulas=400_000):
        """Resolve (YtY + reg I + alpha Y_r^T Y_r) x_r = (1+alpha) Y_r^T 1 para cada linha r.

        Linhas são agrupadas por tamanho de histórico e preenchidas (padding) em blocos,
        o que evita materializar um produto externo por interação.
        """
        f = Y.shape[1]
        YtY = Y.T @ Y + reg * np.eye(f)
        out = np.zeros((R.shape[0], f))
        tam = np.diff(R.indptr)
        ordem = np.argsort(tam, kind='stable')
        ini = 0
        while ini < len(ordem):
            L = max(int(tam[ordem[ini]]), 1)
            n = max(1, celulas // L)
            # garante que o maior histórico do bloco não exceda o orçamento de células
            fim = ini + n
            while fim > ini + 1 and tam[ordem[min(fim, len(ordem)) - 1]] * (fim - ini) > celulas:
                fim = ini + max(1, (fim - ini) // 2)
            linhas = ordem[ini:fim]
            Lmax = max(int(tam[linhas].max()), 1)
            Yp = np.zeros((len(linhas), Lmax, f))
            for j, r in enumerate(linhas):
                idx = R.indices[R.indptr[r]:R.indptr[r + 1]]
                Yp[j, :len(idx)] = Y[idx]
            A = YtY[None] + alpha * np.einsum('nlf,nlg->nfg', Yp, Yp)
            b = (1 + alpha) * Yp.sum(axis=1)
            out[linhas] = np.linalg.solve(A, b[..., None])[..., 0]
            ini = fim
        return out

    def escores(self, us):
        return self.U[us] @ self.V.T


class Hibrido:
    """Mistura convexa por usuário: escore normalizado do modelo + beta * popularidade normalizada."""
    nome = 'ItemKNN+Pop'

    def __init__(self, base, beta: float):
        self.base = base; self.M = base.M; self.beta = beta
        self.popn = self.M.pop / self.M.pop.max()

    def escores(self, us):
        s = self.base.escores(us)
        mx = s.max(axis=1, keepdims=True); mx[mx <= 0] = 1.0
        return s / mx + self.beta * self.popn[None, :]


# --------------------------------------------------------------------------
# Avaliação
# --------------------------------------------------------------------------
def avaliar(modelo, M: Matriz, parte: pd.DataFrame, bloqueio: dict | None = None,
            lote: int = 1024, guardar_listas=False):
    """Avalia top-10 por usuário. 'parte' pode ter vários alvos por usuário.

    bloqueio: dict usuário -> conjunto de índices de itens rejeitados conhecidos.
    Empates: escore, depois popularidade no treino, depois ID numérico crescente.
    """
    alvos = parte.groupby('user').item.apply(list)
    alvos = alvos[alvos.index.isin(M.um)]
    usuarios = list(alvos.index)
    us = np.array([M.um[u] for u in usuarios])
    n_itens = len(M.items); idx = np.arange(n_itens)
    linhas = []; listas = []; rejeitados_expostos = 0
    for ini in range(0, len(us), lote):
        bloco = us[ini:ini + lote]
        S = modelo.escores(bloco).astype(float)
        for n, u in enumerate(bloco):
            nome = usuarios[ini + n]
            a = S[n]
            conhecidos = M.X[u].indices
            a[conhecidos] = -np.inf
            rej = bloqueio.get(nome, ()) if bloqueio is not None else ()
            if len(rej):
                a[list(rej)] = -np.inf
            rank = np.lexsort((idx, -M.pop, -a))[:TOP]
            if bloqueio is None and _rejeitados is not None:
                rejeitados_expostos += int(bool(set(rank) & _rejeitados.get(nome, set())))
            rel = [M.im.get(i, -1) for i in alvos.iloc[ini + n]]
            rel_set = {r for r in rel if r >= 0}
            n_rel = len(rel)
            acertos = np.array([r in rel_set for r in rank], dtype=float)
            h = acertos.sum()
            dcg = float((acertos / np.log2(np.arange(2, TOP + 2))).sum())
            idcg = float((1 / np.log2(np.arange(2, min(n_rel, TOP) + 2))).sum())
            pri = np.flatnonzero(acertos)
            linhas.append((nome, h / TOP, h / n_rel, dcg / idcg,
                           1 / (pri[0] + 1) if len(pri) else 0.0, h > 0,
                           n_rel, n_rel - len(rel_set), len(conhecidos)))
            listas.append(rank)
    per = pd.DataFrame(linhas, columns=['user', 'precision10', 'recall10', 'ndcg10',
                                        'mrr10', 'hit10', 'n_rel', 'alvos_frios', 'hist'])
    plano = np.concatenate(listas)
    m = {k: float(per[k].mean()) for k in ['precision10', 'recall10', 'ndcg10', 'mrr10', 'hit10']}
    m.update(exposicao(plano, M), usuarios=len(per),
             alvos_frios=int(per.alvos_frios.sum()), alvos=int(per.n_rel.sum()))
    if bloqueio is None and _rejeitados is not None:
        m['listas_com_rejeitado'] = rejeitados_expostos / len(per)
    return (m, per, listas) if guardar_listas else (m, per)


_rejeitados = None  # dict usuário -> índices rejeitados, definido pelo script para auditoria


def definir_rejeitados(d):
    global _rejeitados
    _rejeitados = d


def exposicao(plano: np.ndarray, M: Matriz) -> dict:
    n = len(M.items)
    cab = set(np.argsort(-M.pop, kind='stable')[:int(np.ceil(0.1 * n))])
    cont = np.bincount(plano, minlength=n).astype(float)
    srt = np.sort(cont)
    gini = float((2 * np.arange(1, n + 1) - n - 1) @ srt / (n * srt.sum()))
    p = M.pop / M.X.shape[0]
    return {'cobertura': float((cont > 0).mean()),
            'participacao_cabeca': float(np.mean([i in cab for i in plano])),
            'gini_exposicao': gini,
            'novidade': float(np.mean(-np.log2(p[plano])))}


def bootstrap_pareado(a, b, n=2000, semente=42):
    """IC95% percentil e p bilateral aproximado para média(a-b), reamostrando usuários."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(semente)
    boot = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(n)])
    lo, hi = np.quantile(boot, [.025, .975])
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(p, 1.0))


def holm(pvals: dict) -> dict:
    itens = sorted(pvals.items(), key=lambda x: x[1]); m = len(itens); out = {}; prev = 0
    for r, (k, p) in enumerate(itens):
        aj = min(1.0, max(prev, (m - r) * p)); out[k] = aj; prev = aj
    return out
