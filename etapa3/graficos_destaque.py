"""Gráficos de destaque (formato 16:9, prontos para slides) a partir das saídas da etapa 3.

Uso: python graficos_destaque.py <pasta_dados> <pasta_resultados_etapa3> <pasta_saida>
Paleta: azul/laranja/água validados (CVD) para RP3β, ItemKNN e ItemKNN+Pop; popularidade em tinta escura;
EASE e iALS em cinza, sempre com rótulo direto (identidade nunca depende só da cor).
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comum as c

D, R, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)

SUP, INK, INK2, MUTED, GRID, BASE = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
AZUL, LARANJA, AGUA = '#2a78d6', '#eb6834', '#1baf7a'
POP, CINZA = '#3d3c39', '#b3b1a9'
COR = {'RP3β': AZUL, 'ItemKNN': LARANJA, 'ItemKNN+Pop': AGUA, 'Popularidade': POP, 'EASE': CINZA, 'iALS': CINZA,
       'Aleatório': CINZA}
AZUL_RAMPA = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
LAR_RAMPA = ['#fde3d6', '#f9c1a6', '#f4996f', '#eb6834', '#c9501f', '#9c3c14', '#6b290d']

plt.rcParams.update({
    'font.family': ['Liberation Sans', 'Arial', 'DejaVu Sans'], 'font.size': 13,
    'figure.facecolor': SUP, 'axes.facecolor': SUP, 'savefig.facecolor': SUP,
    'axes.edgecolor': BASE, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': .8, 'axes.axisbelow': True, 'axes.titlesize': 14, 'axes.titlecolor': INK,
    'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'legend.frameon': False,
    'legend.labelcolor': INK2, 'xtick.labelsize': 11.5, 'ytick.labelsize': 11.5})


def br(v, d=3):
    return f'{v:.{d}f}'.replace('.', ',').replace('-', '−')


def fmt(d):
    return FuncFormatter(lambda v, p: br(v, d))


def pctf(d=0):
    return FuncFormatter(lambda v, p: f'{v:.{d}f}%'.replace('.', ','))


def cabecalho(fig, titulo, sub):
    fig.text(.04, .955, titulo, fontsize=21, fontweight='bold', color=INK, va='top')
    fig.text(.04, .895, sub, fontsize=13, color=INK2, va='top')


def rodape(fig, txt='Fonte: elaboração própria a partir da coleção Steam v1 (UCSD). Projeto Aplicado III – Etapa 3.'):
    fig.text(.04, .025, txt, fontsize=9.5, color=MUTED)


def salvar(fig, nome):
    fig.savefig(OUT / nome, dpi=200); plt.close(fig)


def ticks_auto(eixo):
    """Casas decimais suficientes para o passo dos ticks (evita 0,0025 aparecer como 0,003)."""
    tk = eixo.get_majorticklocs(); passo = np.min(np.diff(tk))
    d = next(d for d in range(0, 6) if abs(round(passo * 10 ** d) - passo * 10 ** d) < 1e-6)
    eixo.set_major_formatter(fmt(d))


T = pd.read_csv(R / 'A_teste.csv'); G = pd.read_csv(R / 'A_validacao_grades.csv')
A = T[~T.ponto.str.contains('bloqueio')]
ag = A.groupby(['familia', 'ponto']).agg(ndcg=('ndcg10', 'mean'), sd=('ndcg10', 'std'),
                                         cob=('cobertura', 'mean'), cab=('participacao_cabeca', 'mean'),
                                         dif=('dif_ndcg_vs_pop', 'mean')).reset_index()
pop = ag[ag.familia == 'Popularidade'].iloc[0]

# ------------------------------------------------------------ 1. Herói: acerto x exposição
fig = plt.figure(figsize=(13.33, 7.5)); ax = fig.add_axes([.08, .13, .86, .66])
rb = ag[(ag.familia == 'RP3β') & (ag.ponto == 'B')].iloc[0]
cabecalho(fig, f'RP3β (B): mesmo acerto da popularidade, {rb.cob / pop.cob:.0f}× mais catálogo exposto',
          'Teste do protocolo A, média de cinco sorteios (barras: desvio-padrão). Ponto A: maior acerto; '
          'ponto B: maior acerto com ≥ 50% de cobertura na validação.')
ax.axhline(pop.ndcg, color=POP, lw=1.2)
ax.text(9, pop.ndcg + .0003, 'nível da popularidade', color=INK2, fontsize=11, ha='left', va='bottom')
rotulos = {('Popularidade', 'A'): ('Popularidade', (2.2, -.0018)), ('ItemKNN', 'A'): ('ItemKNN (etapa 2)', (-1, -.0016)),
           ('ItemKNN+Pop', 'A'): ('ItemKNN+Pop (A)', (-10.2, .0005)), ('ItemKNN+Pop', 'B'): ('ItemKNN+Pop (B)', (-12.5, -.0012)),
           ('RP3β', 'A'): ('RP3β (A)', (1.2, .0011)), ('RP3β', 'B'): ('RP3β (B)', (1.2, .0012)),
           ('EASE', 'A'): ('EASE', (1.2, -.0006)), ('iALS', 'A'): ('iALS', (1.6, .0012))}
for _, r in ag[(ag.familia != 'Aleatório') & ~((ag.familia == 'ItemKNN') & (ag.ponto == 'B'))].iterrows():
    cor = COR[r.familia]; x = r.cob * 100
    ax.errorbar(x, r.ndcg, yerr=r.sd, fmt='none', ecolor=cor, elinewidth=1.4, capsize=0, alpha=.55, zorder=2)
    ax.scatter(x, r.ndcg, s=150 if r.familia in ('RP3β', 'ItemKNN', 'Popularidade') else 90,
               marker='o' if r.ponto == 'A' else 's', color=cor, edgecolor=SUP, linewidth=2, zorder=3)
    nome, (dx, dy) = rotulos[(r.familia, r.ponto)]
    ax.text(x + dx, r.ndcg + dy, nome, fontsize=12, color=INK, fontweight='bold' if r.familia == 'RP3β' else None)
ra = ag[(ag.familia == 'RP3β') & (ag.ponto == 'A')].iloc[0]
ax.annotate('', xy=(rb.cob * 100 - 1.2, rb.ndcg + .00015), xytext=(ra.cob * 100 + 1.2, ra.ndcg - .0002),
            arrowprops=dict(arrowstyle='-|>', color=AZUL, lw=1.6, connectionstyle='arc3,rad=-.15'))
ax.text((ra.cob + rb.cob) * 50, (ra.ndcg + rb.ndcg) / 2 + .0042, 'ajuste de β:\n+29 p.p. de cobertura',
        color=AZUL, fontsize=11.5, ha='center')
ki = ag[(ag.familia == 'ItemKNN')].iloc[0]
ax.text(ki.cob * 100 - 1, ki.ndcg - .0042, 'mesma cobertura,\nmas acerto abaixo da popularidade\nnos cinco sorteios',
        color=LARANJA, fontsize=11, ha='right')
ax.set(xlim=(-2, 62), ylim=(.1145, .1375), xlabel='Cobertura do catálogo nas listas de teste',
       ylabel='NDCG@10 no teste')
ax.xaxis.set_major_formatter(pctf()); ax.yaxis.set_major_formatter(fmt(3))
rodape(fig); salvar(fig, 'g1_acerto_exposicao.png')

# ------------------------------------------------------------ 2. Concentração da exposição (semente 42)
df, _ = c.carregar(D / 'interacoes.csv.gz'); posd = df[df.positive]
P = c.particao_aleatoria(posd, 42); M = c.matriz(P.treino); C = (M.X.T @ M.X).toarray()
s42 = A[A.semente == 42].set_index(['familia', 'ponto'])
hp = lambda f, p='A': json.loads(s42.loc[(f, p)].hp)
mods = {'Popularidade': c.Popularidade(M),
        'ItemKNN (etapa 2)': c.ItemKNN(M, **hp('ItemKNN'), C=C),
        'RP3β (A)': c.RP3beta(M, **hp('RP3β')),
        'RP3β (B)': c.RP3beta(M, **hp('RP3β', 'B'))}
curvas = {}
for nome, mod in mods.items():
    _, _, listas = c.avaliar(mod, M, P.teste, guardar_listas=True)
    cont = np.sort(np.bincount(np.concatenate(listas), minlength=len(M.items)))[::-1].astype(float)
    curvas[nome] = np.concatenate([[0], np.cumsum(cont) / cont.sum()])
xs = np.linspace(0, 100, len(M.items) + 1); xs[0] = 1e-9
fig = plt.figure(figsize=(13.33, 7.5)); ax = fig.add_axes([.08, .13, .62, .66])
cabecalho(fig, 'Quem aparece nas listas? A popularidade recomenda os mesmos 15 jogos a todos',
          'Fração acumulada das 26.330 posições recomendadas (2.633 usuários × 10), com os jogos ordenados do '
          'mais ao menos exposto. Semente 42.')
estilo = {'Popularidade': (POP, 2.4), 'ItemKNN (etapa 2)': (LARANJA, 2), 'RP3β (A)': ('#86b6ef', 2),
          'RP3β (B)': (AZUL, 2.6)}
for nome, y in curvas.items():
    ax.plot(xs[1:], y[1:] * 100, color=estilo[nome][0], lw=estilo[nome][1], solid_capstyle='round')
gx = np.logspace(-1.6, 2, 200); ax.plot(gx, gx, color=BASE, lw=1)
ax.text(14, 9, 'exposição perfeitamente igual', color=MUTED, fontsize=10.5)
ax.set_xscale('log')
ax.set(xlim=(.03, 100), ylim=(0, 101), xlabel='Jogos do catálogo, do mais ao menos recomendado (escala log)',
       ylabel='Posições recomendadas acumuladas')
ax.set_xticks([.1, 1, 10, 100]); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: (f'{v:g}%').replace('.', ',')))
ax.yaxis.set_major_formatter(pctf())
lax = fig.add_axes([.73, .13, .25, .66]); lax.axis('off')
y0 = .92
for nome, y in curvas.items():
    top1 = np.interp(1, xs, y) * 100; usados = (np.diff(y) > 0).sum() / len(M.items) * 100
    lax.plot([0, .09], [y0, y0], color=estilo[nome][0], lw=4, solid_capstyle='round', transform=lax.transAxes)
    lax.text(.13, y0, nome, transform=lax.transAxes, va='center', fontsize=13, fontweight='bold', color=INK)
    lax.text(.13, y0 - .065, f'top 1% dos jogos: {br(top1, 0)}% das posições', transform=lax.transAxes,
             va='center', fontsize=11, color=INK2)
    lax.text(.13, y0 - .12, f'catálogo exposto: {br(usados, 1)}%', transform=lax.transAxes,
             va='center', fontsize=11, color=INK2)
    y0 -= .24
rodape(fig); salvar(fig, 'g2_concentracao_exposicao.png')

# ------------------------------------------------------------ 3. Diferenças por sorteio
fig = plt.figure(figsize=(13.33, 7.5)); ax = fig.add_axes([.17, .13, .79, .66])
cabecalho(fig, 'Ganho pequeno e consistente, mas não confirmado após a correção de Holm',
          'Diferença de NDCG@10 em relação à popularidade, por sorteio (IC95% bootstrap pareado). '
          'Ponto cheio: IC exclui zero; ◆: média dos cinco sorteios.')
ordem = [('iALS', 'A'), ('EASE', 'A'), ('RP3β', 'B'), ('RP3β', 'A'), ('ItemKNN+Pop', 'B'), ('ItemKNN+Pop', 'A'),
         ('ItemKNN', 'A')]
ax.axvspan(-.02, 0, color=LARANJA, alpha=.035, lw=0); ax.axvspan(0, .02, color=AZUL, alpha=.035, lw=0)
for i, (f, p) in enumerate(ordem):
    s = A[(A.familia == f) & (A.ponto == p)].sort_values('semente')
    cor = COR[f] if f not in ('EASE', 'iALS') else '#8f8d86'
    for k, (_, r) in enumerate(s.iterrows()):
        y = i + (k - 2) * .13
        ax.plot([r.ic95_inf, r.ic95_sup], [y, y], color=cor, lw=1.6, alpha=.75, solid_capstyle='round')
        cheio = r.ic95_inf > 0 or r.ic95_sup < 0
        ax.scatter(r.dif_ndcg_vs_pop, y, s=46, color=cor if cheio else SUP, edgecolor=cor, linewidth=1.6, zorder=3)
    ax.scatter(s.dif_ndcg_vs_pop.mean(), i + .42, marker='D', s=46, color=INK, zorder=3)
    pos = (s.dif_ndcg_vs_pop > 0).sum(); hol = ((s.ic95_inf > 0) & (s.p_holm < .05)).sum()
    neg = (s.ic95_sup < 0).sum()
    txt = f'{pos}/5 positivos · {hol}/5 após Holm' if s.dif_ndcg_vs_pop.mean() > 0 else f'IC abaixo de zero em {neg}/5'
    ax.text(.0135, i, txt, va='center', fontsize=10.5, color=INK2)
ax.axvline(0, color=INK, lw=1.2)
ax.set_yticks(range(len(ordem))); ax.set_yticklabels([f'{f} ({p})' if f not in ('ItemKNN',) else 'ItemKNN (etapa 2)'
                                                     for f, p in ordem], fontsize=12.5, color=INK)
ax.set(xlim=(-.0145, .0185), ylim=(-.6, len(ordem) - .1), xlabel='Δ NDCG@10 (modelo − popularidade)')
ax.xaxis.set_major_formatter(fmt(3)); ax.grid(axis='y', visible=False)
ax.text(-.0005, len(ordem) - .35, '← pior que a popularidade', color=LARANJA, fontsize=11.5, ha='right')
ax.text(.0005, len(ordem) - .35, 'melhor que a popularidade →', color=AZUL, fontsize=11.5)
rodape(fig); salvar(fig, 'g3_diferencas_por_sorteio.png')

# ------------------------------------------------------------ 4. Mapas de calor das grades
fig = plt.figure(figsize=(13.33, 7.5))
cabecalho(fig, 'O ItemKNN depende de λ; no RP3β, β troca acerto por cobertura',
          'Médias de validação nos cinco sorteios. Moldura: ItemKNN mais escolhido no ponto A; α e β do RP3β consolidado.')
cm_az = LinearSegmentedColormap.from_list('az', AZUL_RAMPA); cm_la = LinearSegmentedColormap.from_list('la', LAR_RAMPA)
k = G[G.familia == 'ItemKNN'].copy(); k[['k', 'lam']] = k.hp.apply(lambda h: pd.Series([json.loads(h)['k'], json.loads(h)['shrink']]))
mk = k.groupby(['k', 'lam']).ndcg10.mean().unstack()
r3 = G[G.familia == 'RP3β'].copy()
r3[['a', 'b', 'kk']] = r3.hp.apply(lambda h: pd.Series([json.loads(h)['alpha'], json.loads(h)['beta'], json.loads(h)['k']]))
r3 = r3[r3.kk == 200]
mr = r3.groupby(['a', 'b']).ndcg10.mean().unstack(); mc = r3.groupby(['a', 'b']).cobertura.mean().unstack()
paineis = [(mk, cm_az, 'ItemKNN · NDCG@10', 'λ (redução por suporte)', 'k (vizinhos)', 3, (100, 50)),
           (mr, cm_az, 'RP3β (k = 200) · NDCG@10', 'β (penalização da popularidade)', 'α', 3, (0.5, 0.0)),
           (mc * 100, cm_la, 'RP3β (k = 200) · cobertura (%)', 'β (penalização da popularidade)', 'α', 0, (0.5, 0.0))]
for j, (m, cm, tit, xl, yl, dec, alvo) in enumerate(paineis):
    ax = fig.add_axes([.06 + j * .315, .15, .27, .6])
    ax.grid(False)
    vals = m.values; vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    ax.imshow(vals, cmap=cm, aspect='auto', vmin=vmin - (vmax - vmin) * .05, vmax=vmax)
    for (ii, jj), v in np.ndenumerate(vals):
        luz = (v - vmin) / (vmax - vmin + 1e-12)
        ax.text(jj, ii, br(v, dec) if dec else f'{v:.0f}', ha='center', va='center', fontsize=10,
                color='#ffffff' if luz > .55 else INK)
    ax.set_xticks(range(m.shape[1])); ax.set_xticklabels([br(x, 2).rstrip('0').rstrip(',') if isinstance(x, float) else
                                                          f'{x:,}'.replace(',', '.') for x in m.columns], fontsize=10.5)
    ax.set_yticks(range(m.shape[0])); ax.set_yticklabels([br(x, 2).rstrip('0').rstrip(',') if isinstance(x, float) else
                                                          str(x) for x in m.index], fontsize=10.5)
    ax.set_title(tit, fontsize=13.5, pad=10); ax.set_xlabel(xl, fontsize=11.5); ax.set_ylabel(yl, fontsize=11.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ii = list(m.index).index(alvo[0]); jj = list(m.columns).index(alvo[1])
    ax.add_patch(plt.Rectangle((jj - .5, ii - .5), 1, 1, fill=False, ec=INK, lw=2))
rodape(fig); salvar(fig, 'g4_mapas_hiperparametros.png')

# ------------------------------------------------------------ 5. Hipótese H-T: calendário
dd = df.copy(); dd['mes'] = dd.data.dt.month
dd['linha'] = np.where(dd.ano_inferido, 'sem ano', dd.data.dt.year.astype(str))
cal = dd.groupby(['linha', 'mes']).size().unstack(fill_value=0).reindex(
    ['2011', '2012', '2013', '2014', '2015', 'sem ano']).reindex(columns=range(1, 13), fill_value=0)
fig = plt.figure(figsize=(13.33, 7.5)); ax = fig.add_axes([.1, .2, .8, .55])
cabecalho(fig, 'As avaliações sem ano param em setembro: indício de que são de 2016, o ano da coleta',
          'Avaliações por mês de publicação. Nos anos completos, todos os meses aparecem; '
          'nos registros sem ano, outubro a dezembro estão vazios.')
ax.grid(False)
vmax = cal.values.max()
ax.imshow(np.log1p(cal.values), cmap=cm_az, aspect='auto', vmin=0, vmax=np.log1p(vmax))
for (i, j), v in np.ndenumerate(cal.values):
    ax.text(j, i, f'{v:,}'.replace(',', '.') if v else '—', ha='center', va='center', fontsize=10.5,
            color='#ffffff' if v > 900 else (MUTED if v == 0 else INK))
meses = ['jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez']
ax.set_xticks(range(12)); ax.set_xticklabels(meses, fontsize=12); ax.xaxis.tick_top()
ax.set_yticks(range(len(cal))); ax.set_yticklabels(list(cal.index), fontsize=12.5, color=INK)
ax.get_yticklabels()[-1].set_fontweight('bold')
for sp in ax.spines.values():
    sp.set_visible(False)
ax.add_patch(plt.Rectangle((8.5, 4.5), 3, 1, fill=False, ec=LARANJA, lw=2.2))
ax.text(11.5, 5.75, 'compatível com coleta encerrada em setembro de 2016 (último registro sem ano: 18/09)', ha='right', va='top',
        fontsize=11.5, color=LARANJA)
rodape(fig, 'Fonte: elaboração própria a partir da coleção Steam v1 (UCSD). 2010 omitido (66 registros). '
            'Cores em escala logarítmica.')
salvar(fig, 'g5_calendario_datas.png')

# ------------------------------------------------------------ 6. Do aleatório ao temporal (slope)
b1 = pd.read_csv(R / 'B-com-inferencia_teste.csv'); b2 = pd.read_csv(R / 'B-somente-datados_teste.csv')
fig = plt.figure(figsize=(13.33, 7.5)); ax = fig.add_axes([.08, .13, .66, .66])
cabecalho(fig, 'Quando o teste respeita o tempo, a vantagem dos modelos desaparece',
          'Diferença de NDCG@10 em relação à popularidade (linha zero) em cada protocolo. '
          'Protocolo A: média de cinco sorteios.')
xs3 = [0, 1, 2]; rot3 = ['Aleatório (A)', 'Temporal\ncom datas inferidas', 'Temporal\nsó datados']
series = [('ItemKNN', 'A', 'ItemKNN (etapa 2)'), ('RP3β', 'A', 'RP3β (A)'), ('RP3β', 'B', 'RP3β (B)'),
          ('ItemKNN+Pop', 'A', 'ItemKNN+Pop (A)'), ('EASE', 'A', 'EASE'), ('iALS', 'A', 'iALS')]
ax.axhline(0, color=POP, lw=1.6)
ax.text(-.12, .0003, 'popularidade (referência)', va='bottom', fontsize=11, color=INK2)
fim = []
for f, p, nome in series:
    y = [ag[(ag.familia == f) & (ag.ponto == p)].dif.iloc[0]]
    for t in (b1, b2):
        y.append(t[(t.familia == f) & (t.ponto == p) & ~t.ponto.str.contains('bloq')].dif_ndcg_vs_pop.iloc[0])
    cor = COR[f] if f not in ('EASE', 'iALS') else CINZA
    lw = 2.8 if f in ('RP3β', 'ItemKNN') and p == 'A' else 1.8
    ax.plot(xs3, y, color=cor, lw=lw, marker='o', ms=8, mec=SUP, mew=2, ls='-' if p == 'A' else (0, (1, 0)))
    fim.append([y[-1], nome, cor])
fim.sort()
ult = -1
for yv, nome, cor in fim:
    yy = max(yv, ult + .0011); ult = yy
    ax.text(2.42, yy, nome, va='center', fontsize=11.5, color=INK, fontweight='bold' if 'RP3β (A)' in nome else None)
    ax.plot([2.04, 2.38], [yv, yy], color=cor, lw=.8)
ax.set_xticks(xs3); ax.set_xticklabels(rot3, fontsize=12, color=INK); ax.set_xlim(-.15, 2.4)
ticks_auto(ax.yaxis); ax.set_ylabel('Δ NDCG@10 vs. popularidade'); ax.grid(axis='x', visible=False)
ax.text(-.1, -.0124, 'nenhuma diferença positiva é significativa no protocolo temporal;\n'
                   'o ItemKNN fica significativamente abaixo nos dois', ha='left', fontsize=11, color=INK2)
rodape(fig); salvar(fig, 'g6_aleatorio_vs_temporal.png')

# ------------------------------------------------------------ 7. Segmentos por histórico
sg = pd.read_csv(R / 'segmentos_s42.csv'); sg = sg[sg.ponto == 'A']
fig = plt.figure(figsize=(13.33, 7.5)); ax = fig.add_axes([.08, .13, .66, .66])
cabecalho(fig, 'Quanto maior o histórico, mais a personalização compensa',
          'NDCG@10 no teste por número de avaliações positivas do usuário no treino (semente 42).')
faixas = ['3', '4-5', '6 ou mais']; xf = [0, 1, 2]
fimS = []
for f in ['Popularidade', 'ItemKNN', 'ItemKNN+Pop', 'RP3β', 'EASE', 'iALS']:
    s = sg[sg.familia == f].set_index('faixa').reindex(faixas)
    cor = COR[f]; lw = 2.8 if f in ('RP3β', 'Popularidade', 'ItemKNN') else 1.8
    ax.plot(xf, s.ndcg10, color=cor, lw=lw, marker='o', ms=8, mec=SUP, mew=2)
    fimS.append([s.ndcg10.iloc[-1], f if f != 'ItemKNN' else 'ItemKNN (etapa 2)', cor])
fimS.sort(); ult = -1
for yv, nome, cor in fimS:
    yy = max(yv, ult + .0014); ult = yy
    ax.text(2.42, yy, nome, va='center', fontsize=11.5, color=INK, fontweight='bold' if nome == 'RP3β' else None)
    ax.plot([2.04, 2.38], [yv, yy], color=cor, lw=.8)
n = sg[sg.familia == 'Popularidade'].set_index('faixa').reindex(faixas).usuarios
ax.set_xticks(xf); ax.set_xticklabels([f'{f} positivos\n({int(u)} usuários)' for f, u in zip(faixas, n)],
                                      fontsize=12, color=INK)
ax.set_xlim(-.15, 2.4); ticks_auto(ax.yaxis); ax.set_ylabel('NDCG@10 (teste)')
ax.grid(axis='x', visible=False)
rodape(fig, 'Fonte: elaboração própria a partir da coleção Steam v1 (UCSD). Resultado descritivo, sem teste de hipótese.')
salvar(fig, 'g7_por_historico.png')
print('ok', sorted(p.name for p in OUT.glob('g*.png')))
