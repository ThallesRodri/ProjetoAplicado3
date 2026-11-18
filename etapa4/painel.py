"""Gera o painel interativo (Plotly) da etapa 4: espaço de configurações em 3D, superfície do RP3β, curvas @K e
diferenças em relação à popularidade.

Uso: python painel.py <resultados_etapa4> <pasta_saida>
Saídas: painel_corpo.html (conteúdo para publicação) e index.html (arquivo completo, abre no navegador).
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

R, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
AG = pd.read_csv(R / 'validacao_media_sementes.csv'); T = pd.read_csv(R / 'teste_modelos_finais.csv')
RES = json.loads((R / 'resumo_etapa4.json').read_text(encoding='utf-8'))
SEG = pd.read_csv(R / 'segmentos_5_sementes.csv')


def hp_txt(fam, hp):
    h = json.loads(hp)
    if fam == 'RP3β':
        return f"α={h['alpha']:g}, β={h['beta']:g}, k={h['k']}"
    if fam == 'EASE':
        return f"λ={h['lam']:g}"
    if fam == 'iALS':
        return f"f={h['fatores']}, λ={h['reg']:g}, α={h['alpha']:g}"
    if fam == 'ItemKNN':
        return f"k={h['k']}, λ={h['shrink']:g}"
    if fam == 'ItemKNN+Pop':
        return f"k={h['k']}, λ={h['shrink']:g}, β={h['beta']:g}"
    if fam == 'Fusão':
        return ', '.join(f'{k} {v:g}' for k, v in h.items() if v)
    return ''


cfg = [{'f': r.familia, 'hp': hp_txt(r.familia, r.hp), 'n': round(r.ndcg, 5), 'c': round(r.cob * 100, 2),
        'g': round(r.gini, 4), 'v': round(r.nov, 3)} for r in AG.itertuples() if r.familia != 'Aleatório']
marc = {}
for k in ('cstar', 'final'):
    d = RES[k]; r = AG[(AG.familia == d['familia']) & (AG.hp == json.dumps(d['hp']))].iloc[0]
    marc[k] = {'f': r.familia, 'hp': hp_txt(r.familia, r.hp), 'n': round(r.ndcg, 5), 'c': round(r.cob * 100, 2),
               'g': round(r.gini, 4)}
r3 = AG[AG.familia == 'RP3β'].copy(); hs = r3.hp.apply(json.loads)
r3['a'] = hs.apply(lambda h: h['alpha']); r3['b'] = hs.apply(lambda h: h['beta']); r3['k'] = hs.apply(lambda h: h['k'])
sup = {}
for k in (50, 200):
    s = r3[(r3.k == k) & r3.a.isin([0.25, 0.5, 0.75, 1.0]) & r3.b.isin([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4])]
    z = s.pivot(index='a', columns='b', values='ndcg'); c = s.pivot(index='a', columns='b', values='cob') * 100
    sup[str(k)] = {'a': list(z.index), 'b': list(z.columns), 'z': np.round(z.values, 5).tolist(),
                   'c': np.round(c.values, 2).tolist()}
vistos = {}; modelos = []
for n in ['Popularidade', 'ItemKNN (etapa 2)', 'Máximo acerto (c*)', 'Modelo final (F)', 'Fusão']:
    t = T[T.modelo == n]; k = (t.familia.iloc[0], t.hp.iloc[0])
    if k in vistos:
        vistos[k]['rot'] += ' = ' + n; continue
    m = {'id': n, 'rot': n, 'hp': hp_txt(*k),
         'ndcgk': [round(t[f'ndcg@{i}'].mean(), 7) for i in range(1, 21)],
         'hitk': [round(t[f'hit@{i}'].mean(), 5) for i in range(1, 21)],
         'ndcg': round(t.ndcg10.mean(), 7), 'hit': round(t.hit10.mean(), 7), 'mrr': round(t.mrr10.mean(), 7),
         'cob': round(t.cobertura.mean() * 100, 5), 'cab': round(t.participacao_cabeca.mean() * 100, 5),
         'gini': round(t.gini_exposicao.mean(), 6), 'nov': round(t.novidade.mean(), 6),
         'sob': round(t.sobreposicao_pop.mean() * 100, 5)}
    if n != 'Popularidade':
        s = t.sort_values('semente')
        m['sem'] = [{'s': int(r.semente), 'd': round(r.dif_pop, 7), 'lo': round(r.ic_inf, 7), 'hi': round(r.ic_sup, 7)}
                    for r in s.itertuples()]
        a = RES['agregado_usuarios'][n]; m['ag'] = {'d': round(a['dif_pop'], 7), 'lo': round(a['ic_inf'], 7), 'hi': round(a['ic_sup'], 7)}
        sg = SEG[SEG.modelo == n]
        m['seg'] = [{'fx': r.faixa, 'u': int(r.usuarios), 'd': round(r.dif_pop, 7), 'lo': round(r.ic_inf, 7), 'hi': round(r.ic_sup, 7)}
                    for r in sg.itertuples()]
    vistos[k] = m; modelos.append(m)
P = {'cfg': cfg, 'marc': marc, 'sup': sup, 'modelos': modelos, 'delta': RES['delta'],
     'lat': round(RES['modelo']['ms_por_usuario_top10'], 2), 'usuarios': int(T.usuarios.iloc[0]),
     'jogos': RES['modelo']['jogos'], 'final_hp': hp_txt(RES['final']['familia'], json.dumps(RES['final']['hp'])),
     'final_fam': RES['final']['familia']}

CORPO = r'''<meta name="claude:viewport-fit-sides" content="cover">
<title>Painel do Recomendador Steam</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap">
<style>
/* Layout: coluna única de leitura; cada bloco traz uma pergunta, o gráfico que a responde e uma frase de leitura. */
:root{--bg:#f6f6f3;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;
 --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#6250d6;--pop:#3d3c39;
 --display:'Space Grotesk','Trebuchet MS',Arial,sans-serif;--body:'Inter',Arial,sans-serif}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#121211;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#8f8d86;--line:#32312e;
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#9085e9;--pop:#c9c7bf;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#121211;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#8f8d86;--line:#32312e;
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#9085e9;--pop:#c9c7bf;color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font-family:var(--body);font-size:16px;line-height:1.5;
 padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:28px 56px}
.wrap{max-width:1040px;margin-inline:auto;display:flex;flex-direction:column;gap:22px}
.eyebrow{font-family:var(--display);font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--ink2)}
h1{font-family:var(--display);font-size:clamp(28px,5vw,42px);line-height:1.08;margin:6px 0 8px;text-wrap:balance}
.lead{color:var(--ink2);max-width:68ch;margin:0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px;min-width:0}
.kpi b{font-family:var(--display);font-size:30px;font-variant-numeric:tabular-nums;display:block;line-height:1.15}
.kpi span{font-size:13px;color:var(--ink2)}
.kpi small{display:block;font-size:12px;color:var(--muted);margin-top:2px}
section{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:20px;display:flex;flex-direction:column;gap:10px;min-width:0}
h2{font-family:var(--display);font-size:21px;margin:0;text-wrap:balance}
.sub{color:var(--ink2);font-size:14.5px;margin:0;max-width:75ch}
.leitura{font-size:14.5px;margin:0;max-width:75ch;border-left:3px solid var(--s1);padding-left:12px}
.plot{width:100%;min-width:0}
.ctrl{display:flex;flex-wrap:wrap;gap:6px}
.ctrl button{font:inherit;font-size:13.5px;font-weight:500;border:1px solid var(--line);background:var(--bg);color:var(--ink2);border-radius:999px;padding:6px 14px;cursor:pointer}
.ctrl button[aria-pressed="true"]{background:var(--ink);color:var(--surface);border-color:var(--ink)}
.ctrl button:focus-visible{outline:2px solid var(--s1);outline-offset:2px}
.tw{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left;white-space:normal}
th{color:var(--ink2);font-weight:600;font-size:12.5px}
td i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:8px}
details summary{cursor:pointer;font-size:13.5px;color:var(--ink2);font-weight:500}
footer{font-size:12.5px;color:var(--muted);max-width:80ch}
</style>
<div class="wrap">
<header>
 <div class="eyebrow">Projeto Aplicado III · Mackenzie · Etapa 4</div>
 <h1>Recomendação de jogos na Steam: acerto, exposição e escolha do modelo</h1>
 <p class="lead">Painel interativo dos experimentos. Gire os gráficos 3D, passe o cursor sobre os pontos para ver cada configuração e alterne as métricas. Todos os números vêm dos arquivos de resultados do repositório.</p>
</header>
<div class="kpis" id="kpis"></div>

<section>
 <h2>Quanto de acerto cada configuração troca por exposição?</h2>
 <p class="sub">Cada ponto é uma configuração avaliada na validação (média de cinco sementes). Eixos: cobertura do catálogo, desigualdade da exposição (Gini) e NDCG@10. Clique na legenda para ocultar famílias.</p>
 <div class="plot" id="g3d"></div>
 <p class="leitura" id="l3d"></p>
</section>

<section>
 <h2>Como α e β moldam o RP3β?</h2>
 <p class="sub">Altura: NDCG@10 de validação. Cor: cobertura do catálogo. O expoente β desconta a popularidade do jogo de destino; valores negativos a reforçam.</p>
 <div class="ctrl" role="group" aria-label="Número de vizinhos"><button id="k50" aria-pressed="true">k = 50</button><button id="k200" aria-pressed="false">k = 200</button></div>
 <div class="plot" id="gsup"></div>
 <p class="leitura">A superfície é quase plana em acerto entre β = −0,2 e β = 0,1, enquanto a cor muda muito: nesse trecho, β desloca a exposição quase sem custo de NDCG@10. A partir de β = 0,2, o acerto cai.</p>
</section>

<section>
 <h2>O resultado depende do tamanho da lista?</h2>
 <p class="sub">Teste do protocolo aleatório, média de cinco sementes, com bloqueio de jogos rejeitados.</p>
 <div class="ctrl" role="group" aria-label="Métrica"><button id="mN" aria-pressed="true">NDCG@K</button><button id="mH" aria-pressed="false">Hit@K</button></div>
 <div class="plot" id="gk"></div>
 <p class="leitura" id="lk"></p>
</section>

<section>
 <h2>As diferenças para a popularidade são distinguíveis de zero?</h2>
 <p class="sub">Diferença de NDCG@10 no teste. Pontos claros: cada semente (IC95% por bootstrap pareado). Losango: intervalo agregado por usuário nas cinco sementes.</p>
 <div class="plot" id="gdif"></div>
 <p class="leitura" id="ldif"></p>
</section>

<section>
 <h2>Para quem a personalização compensa?</h2>
 <p class="sub">Diferença de NDCG@10 para a popularidade por número de avaliações positivas do usuário no treino (agregado de cinco sementes, IC95%).</p>
 <div class="plot" id="gseg"></div>
</section>

<section>
 <h2>Tabela dos modelos escolhidos</h2>
 <p class="sub">Teste do protocolo aleatório, média de cinco sementes. Cabeça: participação dos 10% de jogos mais populares do treino. Novidade em bits. Em comum: parte da lista que coincide com a de mais populares.</p>
 <div class="tw"><table id="tab"></table></div>
</section>
<footer id="rod"></footer>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"></script>
<script>
const P = /*DADOS*/null/*FIM*/;
(function(){
'use strict';
const $ = id => document.getElementById(id);
const br = (v, d) => v.toLocaleString('pt-BR', {minimumFractionDigits: d, maximumFractionDigits: d});
const sg = (v, d) => (v > 0 ? '+' : v < 0 ? '−' : '') + br(Math.abs(v), d);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const FAM = {'RP3β': '--s1', 'ItemKNN': '--s2', 'ItemKNN+Pop': '--s3', 'EASE': '--s4', 'iALS': '--s5', 'Fusão': '--s6', 'Popularidade': '--pop'};
const MOD = {'Popularidade': '--pop', 'ItemKNN (etapa 2)': '--s2', 'Máximo acerto (c*)': '--s4', 'Modelo final (F)': '--s1', 'Fusão': '--s6'};
const M = P.modelos, pop = M.find(m => m.id === 'Popularidade'), fin = M.find(m => m.rot.includes('Modelo final (F)'));
const estreito = () => window.innerWidth < 640;
function base(extra){ const ink = css('--ink'), ink2 = css('--ink2'), line = css('--line');
  return Object.assign({paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: {family: 'Inter, Arial, sans-serif', color: ink2, size: 13},
    margin: {l: 64, r: 16, t: 10, b: 52}, hoverlabel: {bgcolor: css('--surface'), bordercolor: line, font: {color: ink, family: 'Inter, Arial, sans-serif'}},
    legend: {orientation: 'h', y: -0.2, font: {color: ink2}}, separators: ',.',
    xaxis: {gridcolor: line, zerolinecolor: ink, linecolor: line, tickfont: {color: css('--muted')}},
    yaxis: {gridcolor: line, zerolinecolor: ink, linecolor: line, tickfont: {color: css('--muted')}}}, extra || {}); }
const CFG = {displayModeBar: false, responsive: true};
let kSup = '50', met = 'ndcgk';

function g3d(){
  const fams = ['RP3β', 'ItemKNN', 'ItemKNN+Pop', 'EASE', 'iALS', 'Fusão', 'Popularidade'], tr = [];
  fams.forEach(f => { const s = P.cfg.filter(c => c.f === f); if(!s.length) return;
    tr.push({type: 'scatter3d', mode: 'markers', name: f, x: s.map(c => c.c), y: s.map(c => c.g), z: s.map(c => c.n),
      text: s.map(c => c.hp || 'referência'), marker: {size: f === 'Popularidade' ? 8 : 5, color: css(FAM[f]), opacity: .85, line: {width: 0}},
      hovertemplate: '<b>' + f + '</b> %{text}<br>cobertura %{x:.1f}%<br>Gini %{y:.3f}<br>NDCG@10 %{z:.4f}<extra></extra>'}); });
  [['cstar', 'c*: maior acerto', '--s4', 'diamond'], ['final', 'F: modelo final', '--s1', 'square']].forEach(([k, n, cor, sym]) => { const m = P.marc[k];
    tr.push({type: 'scatter3d', mode: 'markers+text', name: n, x: [m.c], y: [m.g], z: [m.n], text: [n.split(':')[0]], textposition: 'top center',
      textfont: {color: css('--ink'), size: 13}, marker: {size: 9, symbol: sym, color: css(cor), line: {color: css('--ink'), width: 2}},
      hovertemplate: '<b>' + n + '</b><br>' + m.f + ' ' + m.hp + '<br>cobertura %{x:.1f}%<br>NDCG@10 %{z:.4f}<extra></extra>'}); });
  const ax = t => ({title: {text: t, font: {size: 12}}, gridcolor: css('--line'), backgroundcolor: 'rgba(0,0,0,0)', color: css('--ink2'), zerolinecolor: css('--line')});
  Plotly.react('g3d', tr, base({height: estreito() ? 430 : 560, margin: {l: 0, r: 0, t: 0, b: 0},
    scene: {xaxis: ax('Cobertura (%)'), yaxis: ax('Gini da exposição'), zaxis: ax('NDCG@10'), aspectmode: 'manual', aspectratio: {x: 1.5, y: .9, z: 1},
      camera: {eye: {x: 1.35, y: -1.75, z: .55}, center: {x: 0, y: 0, z: -.12}}, domain: {x: [0, 1], y: [estreito() ? .24 : .08, 1]}},
    legend: {orientation: 'h', y: 0, yanchor: 'bottom', font: {color: css('--ink2'), size: estreito() ? 11 : 13}}}), {displayModeBar: false, responsive: true});
}
function gsup(){
  const s = P.sup[kSup];
  Plotly.react('gsup', [{type: 'surface', x: s.b, y: s.a, z: s.z, surfacecolor: s.c, customdata: s.c, cmin: 0, cmax: 65,
    colorscale: [[0, '#cde2fb'], [.35, '#6da7ec'], [.7, '#256abf'], [1, '#0d366b']],
    colorbar: {title: {text: 'Cobertura (%)', side: 'right'}, thickness: 12, len: .6, tickfont: {color: css('--muted')}, titlefont: {color: css('--ink2')}},
    contours: {x: {show: true, color: 'rgba(255,255,255,.55)', width: 1}, y: {show: true, color: 'rgba(255,255,255,.55)', width: 1}},
    hovertemplate: 'β %{x}<br>α %{y}<br>NDCG@10 %{z:.4f}<br>cobertura %{customdata:.1f}%<extra></extra>'}],
    base({height: estreito() ? 400 : 520, margin: {l: 0, r: 0, t: 0, b: 0}, scene: {
      xaxis: {title: {text: 'β'}, gridcolor: css('--line'), color: css('--ink2'), backgroundcolor: 'rgba(0,0,0,0)'},
      yaxis: {title: {text: 'α'}, gridcolor: css('--line'), color: css('--ink2'), backgroundcolor: 'rgba(0,0,0,0)'},
      zaxis: {title: {text: 'NDCG@10'}, gridcolor: css('--line'), color: css('--ink2'), backgroundcolor: 'rgba(0,0,0,0)'},
      camera: {eye: {x: -1.5, y: -1.7, z: .8}}}}), CFG);
}
function gk(){
  const ks = Array.from({length: 20}, (_, i) => i + 1);
  Plotly.react('gk', M.map(m => ({type: 'scatter', mode: 'lines', name: m.rot, x: ks, y: m[met],
    line: {color: css(MOD[m.id]), width: m.id === 'Popularidade' ? 2.5 : 2, dash: m.id === 'Popularidade' ? 'dash' : 'solid'},
    hovertemplate: 'K = %{x}: %{y:.4f}<extra>' + m.rot + '</extra>'})),
    base({height: estreito() ? 440 : 420, hovermode: 'x unified', margin: {l: 58, r: 12, t: 10, b: estreito() ? 150 : 96}, legend: {orientation: 'h', y: estreito() ? -0.3 : -0.26, font: {color: css('--ink2')}},
      xaxis: Object.assign(base().xaxis, {title: {text: 'Tamanho da lista (K)'}, dtick: 5}),
      yaxis: Object.assign(base().yaxis, {title: {text: met === 'ndcgk' ? 'NDCG@K' : 'Hit@K'}, tickformat: ',.2f'}),
      shapes: [{type: 'line', x0: 10, x1: 10, yref: 'paper', y0: 0, y1: 1, line: {color: css('--muted'), width: 1, dash: 'dot'}}]}), CFG);
}
function gdif(){
  const ms = M.filter(m => m.sem), tr = [];
  ms.forEach((m, i) => { const cor = css(MOD[m.id]);
    tr.push({type: 'scatter', mode: 'markers', name: m.rot, showlegend: false, x: m.sem.map(s => s.d), y: m.sem.map((s, k) => i + (k - 2) * .1 - .08),
      error_x: {type: 'data', symmetric: false, array: m.sem.map(s => s.hi - s.d), arrayminus: m.sem.map(s => s.d - s.lo), color: cor, thickness: 1.3, width: 0},
      marker: {color: cor, size: 6, opacity: .75}, text: m.sem.map(s => 'semente ' + s.s),
      hovertemplate: '%{text}: %{x:.4f}<extra>' + m.rot + '</extra>'});
    tr.push({type: 'scatter', mode: 'markers', showlegend: false, x: [m.ag.d], y: [i + .33],
      error_x: {type: 'data', symmetric: false, array: [m.ag.hi - m.ag.d], arrayminus: [m.ag.d - m.ag.lo], color: css('--ink'), thickness: 2.6, width: 0},
      marker: {color: css('--ink'), size: 10, symbol: 'diamond'},
      hovertemplate: 'agregado: %{x:.4f} (IC95% ' + br(m.ag.lo, 4) + ' a ' + br(m.ag.hi, 4) + ')<extra>' + m.rot + '</extra>'}); });
  Plotly.react('gdif', tr, base({height: 110 + ms.length * 78, margin: {l: estreito() ? 120 : 200, r: 16, t: 10, b: 52},
    xaxis: Object.assign(base().xaxis, {title: {text: estreito() ? 'Δ NDCG@10' : 'Δ NDCG@10 (modelo − popularidade)'}, tickformat: ',.3f', zeroline: true, zerolinewidth: 1.5, nticks: estreito() ? 4 : 8, tickangle: 0}),
    yaxis: Object.assign(base().yaxis, {tickvals: ms.map((_, i) => i + .1), ticktext: ms.map(m => m.rot.replace(' = ', '<br>= ')), showgrid: false, zeroline: false, range: [-.6, ms.length - .3],
      tickfont: {color: css('--ink'), size: 12.5}})}), CFG);
}
function gseg(){
  const ms = M.filter(m => m.seg), fx = ms[0].seg.map(s => estreito() ? s.fx.replace(' ou mais', '+') + '<br>(' + s.u + ')' : s.fx + ' positivos no treino<br>(' + s.u + ' usuários)');
  Plotly.react('gseg', ms.map((m, k) => ({type: 'scatter', mode: 'markers', name: m.rot, x: fx.map((_, i) => i + (k - (ms.length - 1) / 2) * .16), y: m.seg.map(s => s.d),
    error_y: {type: 'data', symmetric: false, array: m.seg.map(s => s.hi - s.d), arrayminus: m.seg.map(s => s.d - s.lo), color: css(MOD[m.id]), thickness: 1.8, width: 0},
    marker: {color: css(MOD[m.id]), size: 10, line: {color: css('--surface'), width: 2}}, hovertemplate: '%{y:.4f}<extra>' + m.rot + '</extra>'})),
    base({height: estreito() ? 440 : 400, margin: {l: 64, r: 12, t: 10, b: estreito() ? 150 : 100}, legend: {orientation: 'h', y: estreito() ? -0.28 : -0.24, font: {color: css('--ink2')}},
      xaxis: Object.assign(base().xaxis, {tickvals: [0, 1, 2], ticktext: fx, tickangle: 0, showgrid: false, zeroline: false, range: [-.5, 2.5]}),
      yaxis: Object.assign(base().yaxis, {title: {text: 'Δ NDCG@10 para a popularidade'}, tickformat: ',.3f', zeroline: true, zerolinewidth: 1.5})}), CFG);
}
function estatico(){
  const k = $('kpis'); k.replaceChildren();
  [[br(fin.ndcg, 4), 'NDCG@10 do modelo final', 'popularidade: ' + br(pop.ndcg, 4)],
   [br(fin.cob, 1) + '%', 'do catálogo aparece nas listas', 'popularidade: ' + br(pop.cob, 1) + '%'],
   [br(P.lat, 2) + ' ms', 'para gerar uma lista de 10 jogos', P.final_fam + ' · ' + P.final_hp],
   [P.usuarios.toLocaleString('pt-BR'), 'usuários avaliados por semente', P.jogos.toLocaleString('pt-BR') + ' jogos no catálogo']].forEach(([v, t, s]) => {
    const d = document.createElement('div'); d.className = 'kpi'; const b = document.createElement('b'); b.textContent = v;
    const a = document.createElement('span'); a.textContent = t; const c = document.createElement('small'); c.textContent = s; d.append(b, a, c); k.append(d); });
  $('l3d').textContent = 'O modelo final (' + P.final_fam + ', ' + P.final_hp + ') é a configuração de maior cobertura cujo acerto, na validação, não é inferior ao da configuração de maior acerto por margem de ' + br(P.delta, 3) + ' em NDCG@10.';
  const d5 = fin.ndcgk[4] - pop.ndcgk[4], d20 = fin.ndcgk[19] - pop.ndcgk[19];
  $('lk').textContent = 'A diferença do modelo final para a popularidade é de ' + sg(d5, 4) + ' em NDCG@5, ' + sg(fin.ndcgk[9] - pop.ndcgk[9], 4) + ' em NDCG@10 e ' + sg(d20, 4) + ' em NDCG@20.';
  $('ldif').textContent = 'Modelo final: ' + sg(fin.ag.d, 4) + ' em NDCG@10 no agregado (IC95% de ' + sg(fin.ag.lo, 4) + ' a ' + sg(fin.ag.hi, 4) + '), com ' + br(fin.cob, 1) + '% de cobertura contra ' + br(pop.cob, 1) + '% da popularidade.';
  const t = $('tab'); t.replaceChildren();
  const cab = ['Modelo', 'Configuração', 'NDCG@10', 'Hit@10', 'MRR@10', 'Cobertura', 'Cabeça', 'Gini', 'Novidade', 'Em comum'];
  const tr = document.createElement('tr'); cab.forEach(c => { const th = document.createElement('th'); th.textContent = c; tr.append(th); }); t.append(tr);
  M.forEach(m => { const r = document.createElement('tr');
    const c0 = document.createElement('td'); const i = document.createElement('i'); i.style.background = 'var(' + MOD[m.id] + ')'; c0.append(i, m.rot); r.append(c0);
    [m.hp || '—', br(m.ndcg, 4), br(m.hit, 3), br(m.mrr, 4), br(m.cob, 1) + '%', br(m.cab, 1) + '%', br(m.gini, 3), br(m.nov, 2), br(m.sob, 1) + '%'].forEach(v => {
      const td = document.createElement('td'); td.textContent = v; r.append(td); }); t.append(r); });
  $('rod').textContent = 'Fonte: elaboração própria a partir da coleção Steam Video Game and Bundle Data, versão 1 (UCSD). Equipe: Leandro Henrique Pereira, Murilo Freitas Rodrigues e Thalles Rodrigues. Curso Superior de Tecnologia em Banco de Dados, Universidade Presbiteriana Mackenzie, 2026.';
}
function tudo(){ g3d(); gsup(); gk(); gdif(); gseg(); }
function par(a, b, f){ $(a).onclick = () => { $(a).setAttribute('aria-pressed', 'true'); $(b).setAttribute('aria-pressed', 'false'); f(true); };
  $(b).onclick = () => { $(b).setAttribute('aria-pressed', 'true'); $(a).setAttribute('aria-pressed', 'false'); f(false); }; }
par('k50', 'k200', p => { kSup = p ? '50' : '200'; gsup(); });
par('mN', 'mH', p => { met = p ? 'ndcgk' : 'hitk'; gk(); });
estatico();
if(window.Plotly){ tudo();
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', tudo);
  let rt, lw = estreito(); addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { if(estreito() !== lw){ lw = estreito(); tudo(); } }, 200); });
  new MutationObserver(tudo).observe(document.documentElement, {attributes: true, attributeFilter: ['data-theme']}); }
else { document.querySelectorAll('.plot').forEach(p => { p.textContent = 'Os gráficos interativos precisam de conexão com a internet para carregar a biblioteca Plotly.'; }); }
})();
</script>
'''
corpo = CORPO.replace('/*DADOS*/null/*FIM*/', json.dumps(P, ensure_ascii=False, separators=(',', ':')))
(OUT / 'painel_corpo.html').write_text(corpo, encoding='utf-8')
(OUT / 'index.html').write_text('<!doctype html>\n<html lang="pt-BR">\n<head>\n<meta charset="utf-8">\n'
                               '<meta name="viewport" content="width=device-width, initial-scale=1">\n</head>\n<body style="margin:0">\n'
                               + corpo + '\n</body>\n</html>\n', encoding='utf-8')
print('ok', len(corpo))
