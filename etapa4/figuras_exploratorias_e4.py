"""Figuras exploratórias e fluxograma da metodologia (independentes dos resultados)."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import comum as c

D = Path(sys.argv[1]); F = Path(sys.argv[2]); F.mkdir(parents=True, exist_ok=True)
AZUL, CINZA, VERM, CLARO = '#2a78d6', '#b3b1a9', '#eb6834', '#cde2fb'
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 11, 'axes.titleweight': 'bold'})


def salvar(fig, nome):
    fig.savefig(F / nome, dpi=220, bbox_inches='tight'); plt.close(fig)


df, aud = c.carregar(D / 'interacoes.csv.gz')
pos = df[df.positive]
info = {}

# Figura: distribuição dos históricos positivos
cont = pos.groupby('user').size()
fig, ax = plt.subplots(figsize=(7.2, 3.3))
ax.hist(cont, bins=np.arange(.5, 11.5), color=AZUL, edgecolor='white')
ax.axvline(4.5, color=VERM, ls='--', lw=1.2, label='Elegibilidade para avaliação (≥ 5 positivos)')
ax.set(xlabel='Avaliações positivas por usuário', ylabel='Usuários', xticks=range(1, 11))
ax.legend(frameon=False)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f'{int(v):,}'.replace(',', '.')))
salvar(fig, 'fig_historicos.png')
info['usuarios_pos'] = int(len(cont)); info['elegiveis'] = int((cont >= 5).sum())
info['hist_media'] = float(cont.mean()); info['hist_mediana'] = float(cont.median())

# Figura: cauda longa
ic = pos.groupby('item').size().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7.2, 3.3))
ax.loglog(np.arange(1, len(ic) + 1), ic.values, color=AZUL, lw=1.6)
n10 = int(np.ceil(.1 * len(ic)))
share = ic.iloc[:n10].sum() / ic.sum()
ax.axvline(n10, color=VERM, ls='--', lw=1)
ax.text(n10 * 1.1, ic.values[0] * 0.5, f'10% mais avaliados\n= {share:.1%} dos positivos'.replace('.', ','), color=VERM, fontsize=9)
ax.set(xlabel='Posição do jogo por popularidade (escala log)', ylabel='Avaliações positivas (escala log)')
salvar(fig, 'fig_cauda_longa.png')
info.update(itens_pos=int(len(ic)), cabeca10_share=float(share), top10_share=float(ic.iloc[:10].sum() / ic.sum()),
            itens_um_positivo=int((ic == 1).sum()))

# Figura: hipótese H-T (datas sem ano)
d = df.copy(); d['ano'] = np.where(d.ano_inferido, 'sem ano', d.data.dt.year.astype(str))
d['iid'] = d.item.astype(int)
ordem = ['2010', '2011', '2012', '2013', '2014', '2015', 'sem ano']
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2))
m = d[d.ano_inferido].data.dt.month.value_counts().reindex(range(1, 13), fill_value=0)
m15 = d[(~d.ano_inferido) & (d.data.dt.year == 2015)].data.dt.month.value_counts().reindex(range(1, 13), fill_value=0)
axs[0].bar(np.arange(1, 13) - .2, m15.values, .4, color=CINZA, label='2015 (datado)')
axs[0].bar(np.arange(1, 13) + .2, m.values, .4, color=AZUL, label='Sem ano')
axs[0].set(xlabel='Mês da publicação', ylabel='Avaliações', xticks=range(1, 13), title='(a) Meses registrados', ylim=(0, 2350))
axs[0].legend(frameon=False, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2)
axs[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f'{int(v):,}'.replace(',', '.')))
dados = [d[d.ano == a].iid.values / 1000 for a in ordem]
bp = axs[1].boxplot(dados, showfliers=False, patch_artist=True, widths=.6)
for b, a in zip(bp['boxes'], ordem):
    b.set_facecolor(AZUL if a == 'sem ano' else CLARO)
for med in bp['medians']:
    med.set_color(VERM)
axs[1].set_xticks(range(1, 8)); axs[1].set_xticklabels(['2010', '11', '12', '13', '14', '15', 'sem\nano'])
axs[1].set(ylabel='ID do jogo (milhares)', title='(b) ID dos jogos avaliados')
fig.tight_layout()
salvar(fig, 'fig_hipotese_datas.png')
info['ht'] = {'meses_sem_ano': m.to_dict(), 'max_id_2015': int(d[d.ano == '2015'].iid.max()),
              'max_id_sem_ano': int(d[d.ano == 'sem ano'].iid.max()),
              'mediana_id': {a: float(np.median(x) * 1000) for a, x in zip(ordem, dados)},
              'frac_id_400k': {a: float((d[d.ano == a].iid >= 400000).mean()) for a in ordem},
              'ultimo_dia_sem_ano': str(d[d.ano_inferido].data.max().date())}
anos = d.ano.value_counts().reindex(ordem).to_dict(); info['por_ano'] = anos

# Figura 1: fluxograma da metodologia (estrutura do template, conteúdo do projeto)
etapas = [
    ('3.1 Definição do problema e objetivos', 'Top-10 de jogos; hipóteses H1 a H3, H-T e critérios'),
    ('3.2 Coleta de dados', 'UCSD Steam v1; SHA-256 dos arquivos; pseudonimização'),
    ('3.3 Pré-processamento e limpeza', 'ausentes; 874 duplicatas; positivos e rejeições; datas (H-T)'),
    ('3.4 Divisão dos dados', 'A: aleatória por usuário, 5 sementes | B: temporal global 80/10/10'),
    ('3.5 Seleção e implementação do algoritmo', 'Popularidade, ItemKNN, ItemKNN+Pop, RP3β, EASE, iALS'),
    ('3.6 Treinamento do modelo', 'matriz esparsa usuário-jogo; estatísticas apenas no treino'),
    ('3.7 Avaliação do desempenho', 'NDCG, Hit, MRR @10; cobertura; Gini; novidade; bootstrap + Holm'),
    ('3.8 Otimização e ajustes', 'grades ampliadas; fusão; regra do modelo final; bloqueio'),
    ('3.9 Implantação e monitoramento', 'modelo persistido; demonstração e oficina; monitoramento'),
]
fig, ax = plt.subplots(figsize=(7.0, 7.8)); ax.set_xlim(0, 10); ax.set_ylim(0, len(etapas) * 1.0 + .2); ax.axis('off')
for i, (t, s) in enumerate(etapas):
    y = len(etapas) - i - .45
    cor = '#fde3d6' if i in (2, 3, 4, 6, 7) else CLARO
    ax.add_patch(FancyBboxPatch((0.4, y - .34), 9.2, .68, boxstyle='round,pad=0.02,rounding_size=0.08',
                                fc=cor, ec='#256abf', lw=1.1))
    ax.text(5, y + .13, t, ha='center', va='center', fontsize=12.5, fontweight='bold', color='#0b0b0b')
    ax.text(5, y - .16, s, ha='center', va='center', fontsize=10.5, color='#333333')
    if i < len(etapas) - 1:
        ax.annotate('', xy=(5, y - .64), xytext=(5, y - .36), arrowprops=dict(arrowstyle='-|>', color=AZUL, lw=1.2))
# laço de realimentação: avaliação -> otimização -> divisão (seleção apenas na validação)
yv = len(etapas) - 7 - .45; yd = len(etapas) - 4 - .45
ax.annotate('', xy=(9.6, yd), xytext=(9.6, yv + .02), arrowprops=dict(arrowstyle='-|>', color=VERM, lw=1.1,
            connectionstyle='arc3,rad=0.35', ls='--'))
ax.text(10.5, (yd + yv) / 2, 'reajuste usa só\na validação;\nteste avaliado\numa única vez', fontsize=9.5, color=VERM,
        va='center', ha='left')
fig.savefig(F / 'fig_fluxograma.png', dpi=220, bbox_inches='tight'); plt.close(fig)
(F / 'info_exploratoria.json').write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str))
print(json.dumps(info, indent=1, ensure_ascii=False, default=str))
