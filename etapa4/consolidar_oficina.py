"""Consolida as respostas do questionário da oficina (arquivos CSV exportados pela demonstração).

Uso: python consolidar_oficina.py <pasta_saida> <respostas1.csv> [respostas2.csv ...]
Saídas: oficina_resumo.csv (por afirmação: n, média, mediana, % de concordância), oficina_resumo.json e
fig_oficina.png (distribuição das respostas). Nenhum dado pessoal é lido: o arquivo só contém jogos escolhidos,
notas de 1 a 5 e comentário opcional.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
df = pd.concat([pd.read_csv(f, sep=';', encoding='utf-8-sig') for f in sys.argv[2:]], ignore_index=True)
ITENS = {'q1': 'Q1 Acerto percebido', 'q2': 'Q2 Novidade', 'q3': 'Q3 Preferência pelo modelo', 'q4': 'Q4 Compreensão',
         'q5': 'Q5 Objetivo extensionista'}
res = []
for q, rot in ITENS.items():
    v = df[q].astype(int)
    res.append({'item': rot, 'n': int(len(v)), 'media': float(v.mean()), 'mediana': float(v.median()),
                'concordam_4_5': float((v >= 4).mean()), 'discordam_1_2': float((v <= 2).mean()),
                **{f'nota_{k}': int((v == k).sum()) for k in range(1, 6)}})
R = pd.DataFrame(res); R.to_csv(OUT / 'oficina_resumo.csv', index=False)
resumo = {'participantes': int(len(df)), 'jogos_escolhidos_media': float(df.jogos_escolhidos.str.count(r'\|').add(1).mean()),
          'jogos_em_comum_media': float(df.jogos_em_comum.mean()), 'fora_da_cabeca_media': float(df.fora_da_cabeca.mean()),
          'comentarios': int(df.comentario.fillna('').str.strip().ne('').sum()), 'itens': res}
(OUT / 'oficina_resumo.json').write_text(json.dumps(resumo, indent=1, ensure_ascii=False), encoding='utf-8')

# Distribuição das respostas: barras empilhadas divergentes (laranja = discordância, cinza = neutro, azul = concordância)
CORES = ['#c9501f', '#f4996f', '#c3c2b7', '#6da7ec', '#256abf']
plt.rcParams.update({'font.family': ['Liberation Sans', 'Arial', 'DejaVu Sans'], 'font.size': 9.5})
fig, ax = plt.subplots(figsize=(7.4, 2.9))
for i, r in enumerate(res[::-1]):
    p = np.array([r[f'nota_{k}'] for k in range(1, 6)]) / max(r['n'], 1) * 100
    esq = -(p[0] + p[1] + p[2] / 2)
    for k in range(5):
        ax.barh(i, p[k], left=esq, color=CORES[k], edgecolor='white', lw=1.2, height=.62)
        if p[k] >= 9:
            ax.text(esq + p[k] / 2, i, f'{p[k]:.0f}%', ha='center', va='center', fontsize=8,
                    color='white' if k in (0, 4) else '#0b0b0b')
        esq += p[k]
ax.axvline(0, color='#0b0b0b', lw=.8)
ax.set_yticks(range(5)); ax.set_yticklabels([r['item'] for r in res[::-1]])
ax.set_xlabel(f'Respostas (%) — n = {len(df)}'); ax.set_xlim(-100, 100)
ax.set_xticks([-100, -50, 0, 50, 100]); ax.set_xticklabels(['100', '50', '0', '50', '100'])
for s in ('top', 'right', 'left'):
    ax.spines[s].set_visible(False)
h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CORES]
ax.legend(h, ['1 discordo totalmente', '2', '3', '4', '5 concordo totalmente'], ncol=5, fontsize=8, frameon=False,
          loc='lower center', bbox_to_anchor=(.5, 1.0), handlelength=1, columnspacing=1)
fig.tight_layout(); fig.savefig(OUT / 'fig_oficina.png', dpi=220, bbox_inches='tight', facecolor='white')
print(R.to_string(index=False)); print(json.dumps({k: v for k, v in resumo.items() if k != 'itens'}, ensure_ascii=False))
