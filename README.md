# Projeto Aplicado III – Recomendação de jogos na Steam

Universidade Presbiteriana Mackenzie – Curso Superior de Tecnologia em Banco de Dados

**Integrantes:** Leandro Henrique Pereira, Murilo Freitas Rodrigues e Thalles Rodrigues

## Objetivo

Avaliar se um sistema de recomendação personalizada de jogos da Steam, treinado com o histórico de avaliações
positivas dos usuários, produz listas de dez títulos mais bem ordenadas que a recomendação por popularidade, e como
essa escolha altera a exposição do catálogo.

O sistema gera, para cada usuário, uma lista top-10 de jogos ainda não avaliados, a partir da coleção pública
*Steam Video Game and Bundle Data* (UCSD, versão 1 – comunidade australiana).

## Links

- **Vídeo de apresentação (YouTube):** https://www.youtube.com/watch?v=V7DzEbDpL8A
- **Vídeo da aplicação na comunidade (oficina):** https://youtu.be/mFnrtEVCAVg?feature=shared
- **Documento final (Etapa 4):** [documentos/Projeto_Aplicado_III_Etapa_4.pdf](documentos/Projeto_Aplicado_III_Etapa_4.pdf)
- **Demonstração interativa:** [demo/index.html](demo/index.html) (baixe e abra no navegador; funciona sem internet)
- **Painel interativo dos resultados:** [painel/index.html](painel/index.html) (precisa de internet para carregar a biblioteca de gráficos)
- **Base de dados:** https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data

## Resultado principal

Teste do protocolo aleatório, média de cinco sorteios, 2.633 usuários por sorteio:

| Modelo | NDCG@10 | Hit@10 | Cobertura do catálogo |
|---|---|---|---|
| Popularidade (referência) | 0,1278 | 0,2224 | 0,5% |
| ItemKNN da etapa 2 | 0,1219 | 0,2100 | 57,0% |
| Fusão EASE + iALS (maior acerto) | 0,1325 | 0,2286 | 5,2% |
| **Modelo final: RP3β (α = 0,5; β = 0,15; k = 50)** | **0,1307** | **0,2258** | **47,1%** |

O modelo final tem acerto não inferior ao do melhor modelo encontrado (margem de 0,005 em NDCG@10) e recomenda
47% do catálogo, contra 0,5% da lista de mais populares. No protocolo temporal, nenhum modelo supera a popularidade
com significância. A análise completa está no documento.

## Estrutura do repositório

```
dados/                 base de trabalho pseudonimizada, nomes dos jogos e manifesto de proveniência (SHA-256)
scripts/               etapa 2: preparação dos dados e prova de conceito (Popularidade x ItemKNN)
etapa3/                etapa 3: protocolos A (aleatório, 5 sementes) e B (temporal), seis famílias de modelos
etapa4/                etapa 4: grades ampliadas, fusão, escolha do modelo final, figuras, painel e demonstração
resultados_etapa3/     saídas da etapa 3 (grades de validação, métricas de teste, segmentos, resumo)
resultados_etapa4/     saídas da etapa 4 (seleção do modelo final, teste, robustez temporal, exemplos, resumo)
modelo_final/          modelo consolidado (cópia de resultados_etapa4/modelo_final gerada pela fase final) (matriz de positivos, pesos item-item, mapas e metadados)
figuras/               figuras da etapa 3
figuras_etapa4/        figuras do documento final
graficos_destaque/     gráficos 16:9 para apresentação
painel/                painel interativo com gráficos 3D
demo/                  demonstração usada na oficina com a comunidade
resultados_oficina/    respostas anônimas da oficina e resumo por afirmação
documentos/            documento final em PDF
```

## Como reproduzir

Requisitos: Python 3.12 ou superior e os pacotes de `requirements.txt` (`pip install -r requirements.txt`).
Execute os comandos **a partir da pasta raiz do repositório**. No Windows (PowerShell):

```powershell
python scripts\executar.py                                        # etapa 2 (NDCG@10 0,124555 / 0,119434)
python etapa3\reproduzir_e2.py dados resultados_etapa3            # confirma a etapa 2
python etapa3\etapa3.py dados resultados_etapa3                   # etapa 3 (cerca de 1 h)

# etapa 4: grade ampliada na validação, uma semente por vez (cerca de 5 min cada)
python etapa4\etapa4.py grade dados resultados_etapa3 resultados_etapa4 42
python etapa4\etapa4.py grade dados resultados_etapa3 resultados_etapa4 7
python etapa4\etapa4.py grade dados resultados_etapa3 resultados_etapa4 13
python etapa4\etapa4.py grade dados resultados_etapa3 resultados_etapa4 2024
python etapa4\etapa4.py grade dados resultados_etapa3 resultados_etapa4 2026
# seleção do modelo final, teste, robustez temporal e exemplos (cerca de 15 min)
python etapa4\etapa4.py final dados resultados_etapa3 resultados_etapa4

python etapa4\figuras_e4.py resultados_etapa4 figuras_etapa4      # figuras novas da etapa 4
python etapa4\figuras_resultados_e4.py resultados_etapa3 figuras_etapa4     # figuras da etapa 3 na mesma paleta
python etapa4\figuras_exploratorias_e4.py dados figuras_etapa4             # estatísticas da base e fluxograma
python etapa4\painel.py resultados_etapa4 painel                  # painel interativo
python etapa4\construir_demo.py modelo_final dados\titulos.csv resultados_etapa4 demo
python etapa4\recomendar_etapa4.py modelo_final dados\titulos.csv   # top-10 de um usuário
```

No Linux ou macOS, troque `\` por `/`. Todos os arquivos são lidos e gravados em UTF-8.
O SHA-256 de `dados/interacoes.csv.gz` deve ser `49959f4c10de06ec090133e36024d00463f3d9f8ab8e2b624e827a7e5fc93f6c`.

## Oficina com a comunidade

1. Abra `demo/index.html` no navegador.
2. Cada participante escolhe de 1 a 10 jogos, compara as duas listas e responde ao questionário anônimo.
3. Ao final, clique em **Baixar respostas (CSV)**.
4. Consolide as respostas: `python etapa4\consolidar_oficina.py resultados_oficina respostas_oficina.csv`

A oficina foi realizada em 12/11/2026, com cinco participantes. As respostas e o resumo estão em `resultados_oficina/`:
quatro dos cinco entenderam a explicação (Q4), o acerto percebido ficou dividido (Q1, média 3,2) e nenhum percebeu
novidade nas sugestões (Q2, média 1,4); em média, seis dos dez jogos da lista personalizada coincidiam com a lista de
populares. A análise está na seção 4.4 do documento.

## Método em resumo

- **Dados:** 59.305 avaliações; 874 duplicatas removidas; 51.692 positivas (sinal de preferência) e 6.739 negativas
  (usadas apenas para bloquear jogos rejeitados). Usuários pseudonimizados por SHA-256.
- **Protocolo A:** usuários com 5 ou mais positivos; um positivo na validação e um no teste; sementes 42, 7, 13,
  2024 e 2026. **Protocolo B:** corte temporal global 80/10/10.
- **Modelos:** Popularidade, Aleatório, ItemKNN, ItemKNN+Pop, RP3β, EASE, iALS e fusão de escores.
- **Seleção:** hiperparâmetros escolhidos apenas na validação; teste lido uma única vez por modelo escolhido.
- **Modelo final:** configuração de maior cobertura cujo NDCG@10 de validação não é inferior ao da melhor
  configuração (limite inferior do IC95% acima de −0,005).
- **Incerteza:** bootstrap pareado de usuários (2.000 reamostras) e correção de Holm.

## Dados e licença

Os dados pertencem aos autores da coleção (McAuley, UCSD; Pathak, Gupta e McAuley, 2017) e são usados para fins
acadêmicos. Textos das avaliações e URLs de perfis não são armazenados.
