"""Figuras dos resultados da etapa 3 (lê as saídas de etapa3.py)."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path(sys.argv[1]); F = Path(sys.argv[2]); F.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': ['Liberation Sans', 'Arial', 'DejaVu Sans'], 'font.size': 9.5,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 10.5, 'axes.titleweight': 'bold'})
CORES = {'Popularidade': '#7f8a96', 'ItemKNN': '#c07a2b', 'ItemKNN+Pop': '#d9b26f', 'RP3β': '#2b4a6f',
         'EASE': '#3f8a6b', 'iALS': '#8a3f6b', 'Aleatório': '#bbbbbb'}
ORDEM = ['Popularidade', 'ItemKNN', 'ItemKNN+Pop', 'RP3β', 'EASE', 'iALS']


from matplotlib.ticker import FuncFormatter, FixedFormatter


def virgula(ax):
    """Separador decimal brasileiro, com casas decimais inferidas dos próprios ticks."""
    for eixo, get in ((ax.xaxis, ax.get_xticks), (ax.yaxis, ax.get_yticks)):
        if eixo.get_scale() != 'linear' or getattr(eixo, '_categorico', False):
            continue
        tk = get()
        dec = 0
        for d in range(0, 5):
            if np.allclose(np.round(tk, d), tk, atol=1e-9):
                dec = d; break
        else:
            dec = 4
        eixo.set_major_formatter(FuncFormatter(lambda v, p, d=dec: f'{v:.{d}f}'.replace('.', ',').replace('-', '−')))


def salvar(fig, nome):
    for ax in fig.axes:
        virgula(ax)
    fig.savefig(F / nome, dpi=220, bbox_inches='tight'); plt.close(fig)


G = pd.read_csv(R / 'A_validacao_grades.csv'); T = pd.read_csv(R / 'A_teste.csv')

# 1) Fronteira acerto x cobertura na validação (semente 42)
g = G[(G.semente == 42) & (G.familia != 'Aleatório')]
fig, ax = plt.subplots(figsize=(7.2, 4.0))
for fam in ORDEM:
    s = g[g.familia == fam]
    ax.scatter(s.cobertura * 100, s.ndcg10, s=22 if fam != 'Popularidade' else 70, alpha=.8,
               color=CORES[fam], label=fam, marker='o' if fam != 'Popularidade' else 'D', edgecolor='white', lw=.4)
pop = g[g.familia == 'Popularidade'].ndcg10.iloc[0]
ax.axhline(pop, color=CORES['Popularidade'], ls='--', lw=1)
ax.axvline(50, color='#a33a2c', ls=':', lw=1)
ax.text(51, ax.get_ylim()[0] + .003, 'cobertura mínima\ndo ponto B (50%)', color='#a33a2c', fontsize=8)
ax.set(xlabel='Cobertura do catálogo nas listas de validação (%)', ylabel='NDCG@10 (validação)',
       ylim=(max(0, g.ndcg10.min() - .005), g.ndcg10.max() + .006))
ax.legend(frameon=False, ncol=3, fontsize=8, loc='lower left')
salvar(fig, 'fig_pareto_validacao.png')

# 2) Diferença de NDCG@10 em relação à popularidade, por semente (ponto A e B)
d = T[(~T.ponto.str.contains('bloqueio')) & (~T.familia.isin(['Popularidade', 'Aleatório']))].copy()
d['rot'] = d.familia + ' (' + d.ponto + ')'
rots = [f'{f} ({p})' for f in ORDEM[1:] for p in ('A', 'B') if f'{f} ({p})' in set(d.rot) and not (f == 'ItemKNN' and p == 'B')]
fig, ax = plt.subplots(figsize=(7.2, 4.6))
sementes = sorted(d.semente.unique())
for i, rt in enumerate(rots[::-1]):
    s = d[d.rot == rt].sort_values('semente')
    for k, (_, row) in enumerate(s.iterrows()):
        y = i + (k - 2) * .12
        sig = (row.ic95_inf > 0) or (row.ic95_sup < 0)
        cor = CORES[row.familia]
        ax.plot([row.ic95_inf, row.ic95_sup], [y, y], color=cor, lw=1.2, alpha=.9)
        ax.plot(row.dif_ndcg_vs_pop, y, 'o', color=cor if sig else 'white', mec=cor, ms=4.5)
    ax.plot(s.dif_ndcg_vs_pop.mean(), i + .38, 'v', color='black', ms=5)
ax.axvline(0, color='black', lw=.8)
ax.set_yticks(range(len(rots))); ax.set_yticklabels(rots[::-1]); ax.yaxis._categorico = True
ax.set_xlabel('Diferença de NDCG@10 em relação à popularidade (teste); IC95% bootstrap pareado')
ax.set_title('Cinco sementes por modelo; círculo cheio: IC95% exclui zero (sem ajuste); triângulo: média',
             fontsize=8.5, fontweight='normal', color='#444444', loc='left')
salvar(fig, 'fig_diferencas_teste.png')

# 3) Acerto x exposição no teste (média das sementes)
a = T[~T.ponto.str.contains('bloqueio') & (T.familia != 'Aleatório')].groupby(['familia', 'ponto']).agg(
    ndcg=('ndcg10', 'mean'), sd=('ndcg10', 'std'), cob=('cobertura', 'mean'), cab=('participacao_cabeca', 'mean')).reset_index()
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.4))
for _, r in a.iterrows():
    for ax, x in ((axs[0], r.cob * 100), (axs[1], r.cab * 100)):
        ax.errorbar(x, r.ndcg, yerr=r.sd, fmt='o' if r.ponto == 'A' else 's', color=CORES[r.familia], ms=5,
                    mfc=CORES[r.familia] if r.ponto == 'A' else 'white', capsize=2, lw=.8)
for ax in axs:
    ax.axhline(a[a.familia == 'Popularidade'].ndcg.iloc[0], color=CORES['Popularidade'], ls='--', lw=.9)
axs[0].set(xlabel='Cobertura do catálogo (%)', ylabel='NDCG@10 médio no teste (± DP)', title='(a) Acerto × cobertura')
axs[1].set(xlabel='Posições ocupadas pelos 10% mais populares (%)', title='(b) Acerto × concentração')
hs = [plt.Line2D([], [], marker='o', ls='', color=CORES[f], label=f) for f in ORDEM]
hs += [plt.Line2D([], [], marker='o', ls='', color='black', label='ponto A'),
       plt.Line2D([], [], marker='s', ls='', mfc='white', color='black', label='ponto B')]
fig.legend(handles=hs, loc='lower center', ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(.5, -.12))
fig.tight_layout()
salvar(fig, 'fig_acerto_exposicao.png')

# 4) Desempenho por faixa de histórico (semente 42, ponto A)
sg = pd.read_csv(R / 'segmentos_s42.csv')
sg = sg[(sg.ponto == 'A') & sg.familia.isin(ORDEM)]
faixas = ['3', '4-5', '6 ou mais']
fig, ax = plt.subplots(figsize=(7.2, 3.3))
w = .13
for k, fam in enumerate(ORDEM):
    s = sg[sg.familia == fam].set_index('faixa').reindex(faixas)
    ax.bar(np.arange(3) + (k - 2.5) * w, s.ndcg10, w, color=CORES[fam], label=fam)
n = sg[sg.familia == 'Popularidade'].set_index('faixa').reindex(faixas).usuarios
ax.xaxis._categorico = True; ax.set_xticks(range(3)); ax.set_xticklabels([f'{f} positivos no treino\n(n = {int(u)})' for f, u in zip(faixas, n)])
ax.set(ylabel='NDCG@10 (teste, semente 42)', ylim=(.10, .14))
ax.legend(frameon=False, ncol=6, fontsize=7.6, loc='upper center', bbox_to_anchor=(.5, 1.12))
salvar(fig, 'fig_segmentos.png')

# 5) Protocolo temporal
linhas = []
for rot in ['B-com-inferencia', 'B-somente-datados']:
    t = pd.read_csv(R / f'{rot}_teste.csv'); t['protocolo'] = rot; linhas.append(t)
TB = pd.concat(linhas)
tb = TB[(TB.ponto == 'A') & TB.familia.isin(ORDEM)]
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=False)
for ax, rot, tit in zip(axs, ['B-com-inferencia', 'B-somente-datados'],
                        ['(a) Com datas inferidas (H-T)', '(b) Somente registros datados']):
    s = tb[tb.protocolo == rot].set_index('familia').reindex(ORDEM)
    ax.bar(range(len(ORDEM)), s.ndcg10, color=[CORES[f] for f in ORDEM])
    for i, (v, lo, hi) in enumerate(zip(s.ndcg10, s.ic95_inf, s.ic95_sup)):
        ax.text(i, v, f'{v:.3f}'.replace('.', ','), ha='center', va='bottom', fontsize=7.5)
    ax.set_xticks(range(len(ORDEM))); ax.set_xticklabels(ORDEM, rotation=35, ha='right', fontsize=8); ax.xaxis._categorico = True
    ax.set_title(tit); ax.set_ylim(0, tb.ndcg10.max() * 1.18)
axs[0].set_ylabel('NDCG@10 (teste temporal)')
fig.tight_layout()
salvar(fig, 'fig_temporal.png')
print('figuras ok')
