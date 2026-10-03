"""Obtém a versão 1 oficial, extrai só campos usados e pseudonimiza usuários.
Não se incluem textos de avaliações ou URLs de perfis na base de trabalho.
"""
from pathlib import Path
from urllib.request import urlretrieve
import ast, gzip, hashlib, json
import pandas as pd
BASE = Path(__file__).resolve().parent
D = BASE / 'dados'
D.mkdir(exist_ok=True)
URLS = {
 'australian_user_reviews.json.gz':
 'https://mcauleylab.ucsd.edu/public_datasets/data/steam/australian_user_reviews.json.gz',
 'steam_games.json.gz':
 'https://cseweb.ucsd.edu/~wckang/steam_games.json.gz'}

def preparar():
    manifesto = {'fontes': URLS, 'acesso': '2026-10-02', 'sha256': {}}
    for nome, url in URLS.items():
        p = D / nome
        if not p.exists():
            urlretrieve(url, p)
        manifesto['sha256'][nome] = hashlib.sha256(p.read_bytes()).hexdigest()
    rows = []; linhas = vazios = 0
    with gzip.open(D / 'australian_user_reviews.json.gz', 'rt') as f:
        for line in f:
            d = ast.literal_eval(line)  # Nunca executar eval em dados externos.
            linhas += 1
            vazios += not bool(d.get('reviews'))
            uid = hashlib.sha256(d['user_id'].encode()).hexdigest()[:20]
            for r in d.get('reviews', []):
                rows.append((uid, r.get('item_id'), r.get('recommend'),
                             r.get('posted', '')))
    df = pd.DataFrame(rows, columns=['user', 'item', 'positive', 'posted'])
    df.to_csv(D / 'interacoes.csv.gz', index=False, compression='gzip')
    jogos = []
    with gzip.open(D / 'steam_games.json.gz', 'rt') as f:
        for line in f:
            r = ast.literal_eval(line)
            if r.get('id'):
                jogos.append((r['id'], r.get('title', r.get('app_name', ''))))
    pd.DataFrame(jogos, columns=['item', 'title']).drop_duplicates('item').to_csv(
        D / 'titulos.csv', index=False)
    manifesto.update(linhas_usuarios=linhas, linhas_sem_reviews=vazios,
                     reviews=len(df), metadados='versão 2; somente exibição de nomes')
    (D / 'proveniencia.json').write_text(json.dumps(manifesto, indent=2), encoding='utf-8')

if __name__ == '__main__':
    preparar()
