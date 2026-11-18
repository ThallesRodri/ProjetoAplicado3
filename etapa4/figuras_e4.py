"""Figuras da etapa 4 (tamanho de documento), a partir das saídas de etapa4.py.

Uso: python figuras_e4.py <resultados_etapa4> <pasta_figuras>
Paleta categórica validada (CVD): azul, laranja, água, amarelo e magenta; popularidade em tinta escura.
Identidade nunca depende só da cor: toda série tem rótulo direto ou legenda.
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap

R, F = Path(sys.argv[1]), Path(sys.argv[2]); F.mkdir(parents=True, exist_ok=True)
INK, INK2, MUTED, GRID, BASE = '#0b0b0b', '#52514e', '#898781', '#e6e5df', '#c3c2b7'
AZUL, LARANJA, AGUA, AMARELO, MAGENTA, VIOLETA, POP = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6', '#3d3c39'
FAM = {'RP3β': AZUL, 'ItemKNN': LARANJA, 'ItemKNN+Pop': AGUA, 'EASE': AMARELO, 'iALS': MAGENTA, 'Fusão': VIOLETA,
       'Popularidade': POP}
MOD = {'Popularidade': POP, 'ItemKNN (etapa 2)': LARANJA, 'Máximo acerto (c*)': AMARELO, 'Modelo final (F)': AZUL,
       'Fusão': VIOLETA}
RAMPA = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
plt.rcParams.update({
    'font.family': ['Liberation Sans', 'Arial', 'DejaVu Sans'], 'font.size': 9.5, 'axes.edgecolor': BASE,
    'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .7, 'axes.axisbelow': True,
    'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlecolor': INK, 'axes.titlelocation': 'left',
    'legend.frameon': False, 'legend.labelcolor': INK2, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5})


def br(v, d=3):
    return f'{v:.{d}f}'.replace('.', ',').replace('-', '−')


def fmt(d):
    return FuncFormatter(lambda v, p: br(v, d))


def pct(d=0):
    return FuncFormatter(lambda v, p: f'{v:.{d}f}%'.replace('.', ','))


def salvar(fig, nome):
    fig.savefig(F / nome, dpi=220, bbox_inches='tight', facecolor='white'); plt.close(fig)


T = pd.read_csv(R / 'teste_modelos_finais.csv')
AG = pd.read_csv(R / 'validacao_media_sementes.csv')
SEL = pd.read_csv(R / 'selecao_modelo_final.csv')
FRONT = pd.read_csv(R / 'fronteira_pareto.csv')
RES = json.loads((R / 'resumo_etapa4.json').read_text(encoding='utf-8'))
G = pd.read_csv(R / 'validacao_grades_completa.csv')
nomes = [n for n in MOD if n in set(T.modelo)]
# modelos com a mesma configuração são desenhados uma só vez
vistos = {}; unicos = []
for n in nomes:
    k = (T[T.modelo == n].familia.iloc[0], T[T.modelo == n].hp.iloc[0])
    if k in vistos:
        vistos[k].append(n)
    else:
        vistos[k] = [n]; unicos.append(n)
ROT = {v[0]: ' = '.join(v) for v in vistos.values()}

# ---------------------------------------------------------------- 1. Curvas @K
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.3))
for ax, met, tit in ((axs[0], 'ndcg', '(a) NDCG@K'), (axs[1], 'hit', '(b) Hit@K (= Recall@K)')):
    fins = []
    for n in unicos:
        t = T[T.modelo == n]
        y = [t[f'{met}@{k}'].mean() for k in range(1, 21)]
        ax.plot(range(1, 21), y, color=MOD[n], lw=2 if n != 'Popularidade' else 2.2, solid_capstyle='round',
                ls='-' if n != 'Popularidade' else (0, (4, 2)), label=ROT[n])
        ax.plot(10, y[9], 'o', ms=4.5, color=MOD[n], mec='white', mew=1)
    ax.axvline(10, color=BASE, lw=.8)
    ax.set_title(tit); ax.set_xlabel('Tamanho da lista (K)'); ax.set_xticks([1, 5, 10, 15, 20])
    ax.yaxis.set_major_formatter(fmt(2)); ax.set_xlim(.5, 20.5)
axs[0].set_ylabel('Média de cinco sementes (teste)')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, fontsize=8.5, bbox_to_anchor=(.5, -.09))
fig.tight_layout(); salvar(fig, 'fig_e4_curvas.png')

# ---------------------------------------------------------------- 2. Seleção do modelo final
el = AG[~AG.familia.isin(['Popularidade', 'Aleatório'])]
pop_val = AG[AG.familia == 'Popularidade'].ndcg.iloc[0]
cs, fn = RES['cstar'], RES['final']
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.5), gridspec_kw={'width_ratios': [1.25, 1]})
ax = axs[0]
for fam in ['ItemKNN', 'ItemKNN+Pop', 'EASE', 'iALS', 'Fusão', 'RP3β']:
    s = el[el.familia == fam]
    ax.scatter(s.cob * 100, s.ndcg, s=13, color=FAM[fam], alpha=.6, lw=0, label=fam, zorder=2)
fr = FRONT.sort_values('cob')
ax.step(fr.cob * 100, fr.ndcg, where='post', color=INK, lw=1.1, zorder=3, label='fronteira de Pareto')
ax.axhline(pop_val, color=POP, lw=1, ls=(0, (4, 2))); ax.text(1, pop_val - .0012, 'popularidade', fontsize=8, color=INK2, va='top')


def linha(d):
    return AG[(AG.familia == d['familia']) & (AG.hp == json.dumps(d['hp']))].iloc[0]


rc, rf = linha(cs), linha(fn)
ax.scatter(rc.cob * 100, rc.ndcg, s=70, marker='D', color=AMARELO, edgecolor=INK, lw=1, zorder=5)
ax.scatter(rf.cob * 100, rf.ndcg, s=80, marker='s', color=AZUL, edgecolor=INK, lw=1, zorder=5)
ax.annotate('c*: maior acerto', (rc.cob * 100, rc.ndcg), xytext=(6, 9), textcoords='offset points', fontsize=8.5, color=INK)
ax.annotate('F: modelo final', (rf.cob * 100, rf.ndcg), xytext=(-8, 11), textcoords='offset points', fontsize=8.5,
            color=INK, ha='right', fontweight='bold')
lo = max(el.ndcg.min(), pop_val - .02)
ax.set(xlabel='Cobertura na validação (%)', ylabel='NDCG@10 na validação (média de 5 sementes)',
       title='(a) Configurações avaliadas', ylim=(lo, el.ndcg.max() + .003))
ax.yaxis.set_major_formatter(fmt(3)); ax.legend(fontsize=7.5, loc='lower left', ncol=2, handletextpad=.3, columnspacing=.8)
ax = axs[1]
s = SEL.sort_values('val_cob')
for _, r in s.iterrows():
    ok = bool(r.nao_inferior); f = (r.familia == fn['familia']) and (r.hp == json.dumps(fn['hp']))
    cor = AZUL if f else (INK2 if ok else LARANJA)
    ax.plot([r.val_cob * 100] * 2, [r.ic_inf, r.ic_sup], color=cor, lw=1.6 if not f else 2.4, solid_capstyle='round')
    ax.scatter(r.val_cob * 100, r.dif_cstar, s=34 if not f else 60, marker='o' if not f else 's',
               color=cor if ok else 'white', edgecolor=cor, lw=1.3, zorder=3)
ax.axhline(0, color=INK, lw=.9); ax.axhline(-RES['delta'], color=LARANJA, lw=1.1, ls=(0, (4, 2)))
ax.text(ax.get_xlim()[0] + .5, -RES['delta'] - .0004, 'margem de não inferioridade (−0,005)', fontsize=8, color=LARANJA, va='top')
ax.set(xlabel='Cobertura da candidata (%)', ylabel='Δ NDCG@10 em relação a c* (IC95%)', title='(b) Candidatas da fronteira')
ax.yaxis.set_major_formatter(fmt(3))
fig.tight_layout(); salvar(fig, 'fig_e4_selecao.png')

# ---------------------------------------------------------------- 3. Ampliação das grades (H4)
def serie(fam, chave, fixos):
    g = AG[AG.familia == fam].copy()
    hs = g.hp.apply(json.loads)
    m = hs.apply(lambda h: all(h.get(k) == v for k, v in fixos.items()))
    g = g[m]; g['x'] = hs[m].apply(lambda h: h[chave])
    return g.sort_values('x')


fx = {'RP3β': RES['componentes_fusao']['RP3β']}
paineis = [('EASE', 'lam', {}, 'λ', 5000, True),
           ('iALS', 'reg', {'alpha': RES['componentes_fusao']['iALS']['alpha']}, 'λ', 3000, True),
           ('ItemKNN+Pop', 'beta', {'k': 100, 'shrink': 50}, 'β', 5, True),
           ('RP3β', 'beta', {'alpha': 0.5, 'k': 50}, 'β', 0, False)]
fig, axs = plt.subplots(2, 4, figsize=(7.4, 3.9), sharex='col')
for j, (fam, ch, fixos, rot, lim, log) in enumerate(paineis):
    s = serie(fam, ch, fixos)
    for i, (col, yl) in enumerate((('ndcg', 'NDCG@10 (validação)'), ('cob', 'Cobertura (%)'))):
        ax = axs[i, j]; y = s[col] * (100 if col == 'cob' else 1)
        ax.plot(s.x, y, color=FAM[fam], lw=1.8, marker='o', ms=3.5, mec='white', mew=.6)
        if log:
            ax.set_xscale('log'); ax.axvspan(lim * 1.3, s.x.max() * 1.6, color=FAM[fam], alpha=.09, lw=0)
            ax.set_xlim(s.x.min() / 1.6, s.x.max() * 1.6)
        else:
            ax.axvspan(s.x.min() - .04, -.03, color=FAM[fam], alpha=.09, lw=0); ax.set_xlim(s.x.min() - .04, s.x.max() + .04)
        if i == 0:
            ax.axhline(pop_val, color=POP, lw=.9, ls=(0, (4, 2))); ax.set_title(fam, fontsize=9.5)
            ax.yaxis.set_major_formatter(fmt(3))
        else:
            ax.set_xlabel(rot); ax.yaxis.set_major_formatter(pct())
        if j == 0:
            ax.set_ylabel(yl, fontsize=8.5)
        ax.tick_params(labelsize=7.5)
fig.tight_layout(w_pad=.6); salvar(fig, 'fig_e4_ampliacao.png')

# ---------------------------------------------------------------- 4. Superfície 3D do RP3β (k = 50)
r3 = AG[AG.familia == 'RP3β'].copy(); hs = r3.hp.apply(json.loads)
r3['a'] = hs.apply(lambda h: h['alpha']); r3['b'] = hs.apply(lambda h: h['beta']); r3['k'] = hs.apply(lambda h: h['k'])
r3 = r3[(r3.k == 50) & r3.a.isin([0.25, 0.5, 0.75, 1.0]) & r3.b.isin([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4])]
Z = r3.pivot(index='a', columns='b', values='ndcg'); Cb = r3.pivot(index='a', columns='b', values='cob')
Bm, Am = np.meshgrid(Z.columns.values, Z.index.values)
cm = LinearSegmentedColormap.from_list('az', RAMPA)
fig = plt.figure(figsize=(7.4, 4.3)); ax = fig.add_subplot(111, projection='3d')
norm = plt.Normalize(0, max(.6, Cb.values.max()))
ax.plot_surface(Bm, Am, Z.values, facecolors=cm(norm(Cb.values)), edgecolor='white', lw=.6, shade=False, antialiased=True)
ax.set_xlabel('β (penalização da popularidade)', labelpad=6); ax.set_ylabel('α', labelpad=4)
ax.set_zlabel('NDCG@10 (validação)', labelpad=10)
ax.zaxis.set_major_formatter(fmt(3)); ax.xaxis.set_major_formatter(fmt(1)); ax.yaxis.set_major_formatter(fmt(2))
ax.view_init(elev=24, azim=-58); ax.tick_params(labelsize=7.5)
sm = plt.cm.ScalarMappable(norm=norm, cmap=cm); sm.set_array([])
cb = fig.colorbar(sm, ax=ax, shrink=.55, pad=.17); cb.set_label('Cobertura do catálogo', fontsize=8.5)
cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v * 100:.0f}%')); cb.outline.set_visible(False)
salvar(fig, 'fig_e4_superficie3d.png')

# ---------------------------------------------------------------- 5. Teste dos modelos escolhidos
ag = RES['agregado_usuarios']
ordem = [n for n in unicos if n != 'Popularidade'][::-1]
fig, ax = plt.subplots(figsize=(7.4, 2.9))
for i, n in enumerate(ordem):
    s = T[T.modelo == n].sort_values('semente')
    for k, (_, r) in enumerate(s.iterrows()):
        y = i + (k - 2) * .1 - .08
        ax.plot([r.ic_inf, r.ic_sup], [y, y], color=MOD[n], lw=1.1, alpha=.65, solid_capstyle='round')
        cheio = r.ic_inf > 0 or r.ic_sup < 0
        ax.scatter(r.dif_pop, y, s=16, color=MOD[n] if cheio else 'white', edgecolor=MOD[n], lw=1, zorder=3)
    a = ag[n]
    ax.plot([a['ic_inf'], a['ic_sup']], [i + .33] * 2, color=INK, lw=2.2, solid_capstyle='round')
    ax.scatter(a['dif_pop'], i + .33, marker='D', s=30, color=INK, zorder=4)
ax.axvline(0, color=INK, lw=.9)
ax.set_yticks(range(len(ordem))); ax.set_yticklabels([ROT[n] for n in ordem], color=INK, fontsize=9)
ax.set_xlabel('Δ NDCG@10 em relação à popularidade (IC95%)'); ax.xaxis.set_major_formatter(fmt(3))
ax.grid(axis='y', visible=False); ax.set_ylim(-.55, len(ordem) - .35)
ax.plot([], [], color=INK, lw=2.2, marker='D', ms=4.5, label='agregado por usuário (5 sementes)')
ax.plot([], [], color=MUTED, lw=1.1, marker='o', ms=3.5, mfc='white', label='cada semente (cheio: IC exclui zero)')
ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.2), ncol=2)
fig.tight_layout(); salvar(fig, 'fig_e4_teste.png')

# ---------------------------------------------------------------- 6. Segmentos por histórico (5 sementes)
S = pd.read_csv(R / 'segmentos_5_sementes.csv')
faixas = ['3', '4-5', '6 ou mais']; ms = [n for n in unicos if n != 'Popularidade']
fig, ax = plt.subplots(figsize=(7.4, 3.7)); w = .8 / len(ms)
for k, n in enumerate(ms):
    s = S[S.modelo == n].set_index('faixa').reindex(faixas); x = np.arange(3) + (k - (len(ms) - 1) / 2) * w
    ax.errorbar(x, s.dif_pop, yerr=[s.dif_pop - s.ic_inf, s.ic_sup - s.dif_pop], fmt='o', color=MOD[n], ms=5.5,
                mec='white', mew=1, elinewidth=1.5, capsize=0, label=ROT[n])
ax.axhline(0, color=INK, lw=.9)
nu = S[S.modelo == 'Popularidade'].set_index('faixa').reindex(faixas).usuarios
ax.set_xticks(range(3)); ax.set_xticklabels([f'{f} positivos no treino\n({int(u)} usuários)' for f, u in zip(faixas, nu)], color=INK)
ax.set_ylabel('Δ NDCG@10 em relação à popularidade'); ax.yaxis.set_major_formatter(fmt(3)); ax.grid(axis='x', visible=False)
ax.legend(fontsize=8.5, ncol=3, loc='lower center', bbox_to_anchor=(.5, 1.0))
fig.tight_layout(); salvar(fig, 'fig_e4_segmentos.png')
print('ok', sorted(p.name for p in F.glob('fig_e4_*.png')))
