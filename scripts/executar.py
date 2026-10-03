"""Prova de conceito Steam - etapa 2. Execução: python executar.py.
Avaliação retrospectiva aleatória; não representa previsão cronológica.
"""
from pathlib import Path
import json, time, platform, hashlib
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix, save_npz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE = Path(__file__).resolve().parent
D, R = BASE/'dados', BASE/'resultados'
R.mkdir(exist_ok=True)
SEED, TOP = 42, 10
if not (D/'interacoes.csv.gz').exists():
    from preparar_dados import preparar
    preparar()
raw = pd.read_csv(D/'interacoes.csv.gz', dtype={'user':str, 'item':str})
assert raw['positive'].isin([True, False]).all()
missing = raw[['user','item','positive']].isna().sum().to_dict()
assert sum(missing.values()) == 0
# Verifica se duplicatas discordam antes de manter a primeira ocorrência.
conflicts = int((raw.groupby(['user','item']).positive.nunique() > 1).sum())
assert conflicts == 0, 'Pares duplicados têm rótulos conflitantes; revisar.'
df = raw.drop_duplicates(['user','item']).copy()
pos = df.loc[df.positive].sort_values(['user','item']).reset_index(drop=True)
counts = pos.groupby('user').size()
eligible = counts[counts >= 5].index
# Um positivo aleatório para validação e outro para teste por usuário elegível.
# Usuários com menos de 5 positivos contribuem apenas para o treino.
rng = np.random.default_rng(SEED)
valid_idx, test_idx = [], []
for user, p in pos.groupby('user', sort=True):
    if user in eligible:
        a, b = rng.choice(p.index.to_numpy(), size=2, replace=False)
        valid_idx.append(a); test_idx.append(b)
valid = pos.loc[valid_idx].sort_values('user')
test = pos.loc[test_idx].sort_values('user')
train = pos.drop(index=valid_idx+test_idx)
pairs = lambda x: set(zip(x.user,x.item))
assert pairs(train).isdisjoint(pairs(valid))
assert pairs(train).isdisjoint(pairs(test))
assert pairs(valid).isdisjoint(pairs(test))
# Vocabulário, popularidade e similaridades usam somente o treino.
users = sorted(train.user.unique()); items = sorted(train.item.unique(),key=int)
um = {x:i for i,x in enumerate(users)}; im = {x:i for i,x in enumerate(items)}
X = csr_matrix((np.ones(len(train),dtype=np.float64),
                ([um[x] for x in train.user],[im[x] for x in train.item])),
               shape=(len(users),len(items)))
pop = np.asarray(X.sum(axis=0)).ravel()
C = (X.T @ X).toarray()
np.fill_diagonal(C,0)
cosine = C / np.sqrt(pop[:,None]*pop[None,:])
start = time.perf_counter()

def fit(k, shrink):
    # Regulariza semelhanças com pouca coocorrência; top-k por item de origem.
    S = cosine * C / (C + shrink) if shrink else cosine.copy()
    if shrink:
        S = np.nan_to_num(S)
    for i in range(len(items)):
        ids = np.lexsort((np.arange(len(items)), -S[i]))[:k]
        mask = np.ones(len(items),dtype=bool); mask[ids]=False
        S[i,mask]=0
    return csr_matrix(S)

def evaluate(part, S=None, return_recs=False):
    us = part.user.map(um).to_numpy()
    scores = np.tile(pop, (len(us),1)) if S is None else (X[us]@S).toarray()
    rows=[]; all_recs=[]
    for n, r in enumerate(part.itertuples()):
        known = X[us[n]].indices
        a = scores[n].copy(); a[known] = -np.inf
        # Similaridade primeiro, popularidade apenas para desempate/fallback.
        rank = np.lexsort((np.arange(len(items)), -pop, -a))[:TOP]
        assert len(rank)==TOP and not set(rank).intersection(known)
        target = im.get(r.item, -1)
        loc = np.flatnonzero(rank == target)
        hit = float(len(loc)>0)
        ndcg = 1/np.log2(loc[0]+2) if hit else 0.0
        mrr = 1/(loc[0]+1) if hit else 0.0
        rows.append((r.user, hit/TOP, hit, ndcg, mrr, target>=0))
        all_recs.append(rank)
    per = pd.DataFrame(rows,columns=['user','precision10','recall10',
                       'ndcg10','mrr10','target_in_train'])
    flat=np.concatenate(all_recs)
    head=set(np.argsort(-pop,kind='stable')[:int(np.ceil(.1*len(items)) )])
    metrics={key:float(per[key].mean()) for key in
             ['precision10','recall10','ndcg10','mrr10']}
    metrics.update(coverage=float(len(np.unique(flat))/len(items)),
       head_share=float(np.mean([i in head for i in flat])),
       users=len(part),cold_targets=int((~per.target_in_train).sum()))
    return metrics,per,all_recs if return_recs else None

# Seleção na validação; teste permanece intocado até a decisão.
experiments=[]
for k in (20,50,100):
    for shrink in (0,10,50):
        S=fit(k,shrink)
        m,_,_=evaluate(valid,S)
        experiments.append({'k':k,'shrink':shrink,**m})
val=pd.DataFrame(experiments).sort_values(['ndcg10','k','shrink'],
                 ascending=[False,True,True])
val.to_csv(R/'validacao.csv',index=False)
best=val.iloc[0]; S=fit(int(best.k),int(best.shrink))
# Sem retreino com validação: comparação usa a mesma matriz de treino.
base,bper,_=evaluate(test)
model,mper,recs=evaluate(test,S,True)
baseval,_,_=evaluate(valid)
metrics=pd.DataFrame([{'modelo':'Popularidade',**base},
                      {'modelo':'ItemKNN',**model}])
metrics.to_csv(R/'metricas_teste.csv',index=False)
mper.to_csv(R/'metricas_por_usuario.csv',index=False)
# Incerteza condicional ao split: bootstrap pareado dos mesmos usuários.
diff=mper.ndcg10.to_numpy()-bper.ndcg10.to_numpy()
brng=np.random.default_rng(SEED)
boot=np.array([brng.choice(diff,size=len(diff),replace=True).mean()
               for _ in range(2000)])
ci=np.quantile(boot,[.025,.975]).tolist()
# Artefatos de auditoria e modelo ajustado.
for nome,data in [('treino',train),('validacao',valid),('teste',test)]:
    data.to_csv(R/f'{nome}_interacoes.csv.gz',index=False,compression='gzip')
save_npz(R/'matriz_treino.npz',X);save_npz(R/'modelo_itemknn.npz',S)
(R/'mapas.json').write_text(json.dumps({'users':users,'items':items}),encoding='utf-8')
# Exemplo determinístico: primeiro usuário da lista de teste, sem seleção por acerto.
r=test.iloc[0];u=um[r.user]
titles=pd.read_csv(D/'titulos.csv',dtype={'item':str}).set_index('item').title.to_dict()
example={'user':r.user,'treino':[{'id':items[i],'titulo':titles.get(items[i],items[i])}
     for i in X[u].indices], 'alvo_teste':{'id':r['item'],'titulo':titles.get(r['item'],r['item'])},
     'top10':[{'id':items[i],'titulo':titles.get(items[i],items[i])} for i in recs[0]]}
(R/'exemplo.json').write_text(json.dumps(example,ensure_ascii=False,indent=2))
# Estatísticas e gráficos exploratórios não são usados para escolher parâmetros.
items_count=pos.groupby('item').size().sort_values(ascending=False)
summary={'raw_reviews':len(raw),'raw_users':raw.user.nunique(),
 'raw_items':raw.item.nunique(),'duplicate_pairs':len(raw)-len(df),
 'conflicting_pairs':conflicts,'missing':missing,'clean_reviews':len(df),
 'positive':len(pos),'negative':int((~df.positive).sum()),
 'positive_users':pos.user.nunique(),'positive_items':pos.item.nunique(),
 'eligible_users':len(eligible),'train_interactions':len(train),
 'train_users':len(users),'train_items':len(items),
 'train_sparsity':float(1-X.nnz/(X.shape[0]*X.shape[1])),
 'user_counts':counts.describe().to_dict(),
 'top10_item_share':float(items_count.head(10).sum()/len(pos)),
 'top1_id':items_count.index[0],'top1_count':int(items_count.iloc[0]),
 'chosen_k':int(best.k),'chosen_shrink':int(best.shrink),
 'base_validation':baseval,'model_validation':float(best.ndcg10),
 'ndcg_difference':float(diff.mean()),'ndcg_difference_ci95':ci,
 'runtime_model_seconds':time.perf_counter()-start,
 'versions':{'python':platform.python_version(),'numpy':np.__version__,
 'pandas':pd.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__},
 'data_hash':hashlib.sha256((D/'interacoes.csv.gz').read_bytes()).hexdigest()}
(R/'resumo.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
    'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
    fig.text(.01,.01,'Fonte: elaboração própria com Steam v1 (UCSD). Prova de conceito, 02/10/2026.',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(R/name,dpi=190);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4));ax.hist(counts,bins=np.arange(.5,11.5),color='#384e69',edgecolor='white');ax.axvline(4.5,color='#a43c32',ls='--',label='Elegibilidade: pelo menos 5');ax.set(xlabel='Avaliações positivas por usuário',ylabel='Usuários',title='Históricos positivos predominantemente curtos');ax.legend();save(fig,'historicos.png')
fig,ax=plt.subplots(figsize=(8,4));ax.plot(np.arange(1,len(items_count)+1),items_count.to_numpy(),color='#384e69');ax.set(xscale='log',yscale='log',xlabel='Posição do jogo por popularidade (log)',ylabel='Avaliações positivas (log)',title='Concentração das interações por jogo');save(fig,'popularidade.png')
fig,axs=plt.subplots(1,3,figsize=(9,3.5))
for ax,key,label in zip(axs,['recall10','ndcg10','coverage'],['Recall@10','NDCG@10','Cobertura']):
 ax.bar(metrics.modelo,metrics[key],color=['#aeb6bd','#384e69']);ax.set_title(label);ax.set_ylim(0,max(metrics[key])*1.25)
 for i,v in enumerate(metrics[key]): ax.text(i,v,f'{v:.3f}',ha='center',va='bottom')
save(fig,'comparacao.png')
print(json.dumps(summary,indent=2));print(metrics.to_string(index=False));print(val[['k','shrink','ndcg10']].to_string(index=False))
