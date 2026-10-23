"""Gera a página estática do curso, no mesmo formato de /curso-yolo/."""

import html
import re
from pathlib import Path

from build_exercicios import lista_exercicios
from comum import colab_url

PORTAL = Path("/Volumes/NAUBER/portal-nauber/public/curso-transfer")
REPO = "https://github.com/naubergois/curso-transfer-colab"
RAIZ = Path(__file__).parent

ROTULO = {1: "Pesos", 2: "Base", 3: "Fotos", 4: "Medida"}

AULAS_PAGINA = [
    (
        "01",
        "O peso que viajou",
        "A ResNet-18 chega sabendo mil classes. A cabeça nova tem duas. O resto fica parado, e a conta mostra quantos pesos ainda podem mudar.",
        "CPU · o peso baixa uma vez · 6 exercícios",
        "notebooks/01-o-peso-que-viajou.ipynb",
        '''rede = resnet18(<b>weights</b>=IMAGENET)

for p in modelo.parameters():
    p.requires_grad = <b>False</b>

modelo.fc = Linear(512, <b>2</b>)''',
    ),
    (
        "02",
        "Congelar e soltar",
        "O último bloco da base volta a treinar, com passo curto. A cabeça anda dez vezes mais rápido. A primeira camada continua travada.",
        "CPU · o peso baixa uma vez · 6 exercícios",
        "notebooks/02-congelar-e-soltar.ipynb",
        '''layer4.requires_grad = <b>True</b>

Adam([
    {"params": base, "lr": <b>1e-4</b>},
    {"params": cabeça, "lr": <b>1e-3</b>},
])''',
    ),
    (
        "03",
        "Poucas fotos",
        "Trinta e duas fotos de círculo e quadrado treinam só a cabeça. A base não anda. O acerto é lido em dezesseis fotos que ficaram de fora.",
        "CPU · cerca de 2 minutos · 6 exercícios",
        "notebooks/03-poucas-fotos.ipynb",
        '''antes = conv1.weight.clone()

# duas épocas, só a cabeça
opt.step()

base_andou = <b>(conv1 - antes).abs().max()</b>''',
    ),
    (
        "04",
        "Medir no seu dado",
        "Um par de matrizes finas fica ao lado do peso grande. A prova mede o seu conjunto, e de novo o mesmo desenho com a cor trocada.",
        "CPU · cerca de 2 minutos · 6 exercícios",
        "notebooks/04-medir-no-seu-dado.ipynb",
        '''delta = <b>B @ A</b>   # posto 4

acerto_seu = prova(seu)
acerto_longe = prova(<b>cor trocada</b>)''',
    ),
]


def inline(texto):
    texto = html.escape(texto)
    texto = re.sub(r"`([^`]+)`", r"<code>\1</code>", texto)
    texto = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", texto)
    return texto


def cartao(i, aula, n, ex, arquivo):
    return f'''    <article class="ex">
      <div class="ex-topo"><span class="ex-num">{i:02d}</span><span class="ex-aula">Aula {aula} · {ROTULO[aula]} · exercício {n}</span></div>
      <h3>{html.escape(ex["titulo"])}</h3>
      <p>{inline(ex["enunciado"].strip())}</p>
      <a class="botao" href="{colab_url("exercicios/" + arquivo)}" target="_blank" rel="noopener">Abrir no Colab</a>
    </article>'''


def aulas_html():
    blocos = []
    for num, titulo, texto, meta, caminho, codigo in AULAS_PAGINA:
        blocos.append(f'''    <article class="aula">
      <div class="midia"><pre>{codigo}</pre></div>
      <div class="corpo">
        <span class="num">AULA {int(num)}</span>
        <h3>{html.escape(titulo)}</h3>
        <p>{html.escape(texto)}</p>
        <div class="meta">{html.escape(meta)}</div>
        <a class="botao" href="{colab_url(caminho)}" target="_blank" rel="noopener">Abrir no Colab</a>
      </div>
    </article>''')
    return "\n\n".join(blocos)


def pagina():
    cards = "\n".join(
        cartao(i, aula, n, ex, arquivo)
        for i, (_nb, aula, n, ex, arquivo) in enumerate(lista_exercicios(), start=1)
    )
    return f'''<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Transfer learning na prática: o peso que já veio | Nauber Gois</title>
<meta name="description" content="Quatro notebooks de transfer learning no Colab: cabeça nova, base congelada, poucas fotos e a medida no seu conjunto. Vinte e quatro exercícios.">
<meta property="og:title" content="Transfer learning na prática, direto no Colab">
<meta property="og:description" content="A rede já viu milhões de fotos. Você troca a cabeça, congela a base e mede no seu dado.">
<style>
  :root {{
    --bg: #0d1117; --card: #161b22; --line: #30363d;
    --text: #e6edf3; --muted: #9da7b3; --accent: #f7c948; --link: #58a6ff;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--text);
    font: 17px/1.6 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }}
  main {{ max-width: 1040px; margin: 0 auto; padding: 56px 20px 80px; }}
  header h1 {{ font-size: clamp(2rem, 5vw, 3rem); line-height: 1.15; margin: 0 0 16px; }}
  header h1 span {{ color: var(--accent); }}
  .lead {{ color: var(--muted); max-width: 720px; font-size: 1.1rem; margin: 0 0 12px; }}
  .hero {{ margin: 32px 0 48px; border-radius: 12px; overflow: hidden; border: 1px solid var(--line); }}
  .hero pre {{
    margin: 0; padding: 28px 24px; color: #c9d1d9;
    font: 16px/1.7 ui-monospace, SFMono-Regular, Menlo, monospace;
    background: linear-gradient(160deg, #1b2230, #0d1117);
  }}
  .hero pre b {{ color: var(--accent); font-weight: 600; }}
  h2 {{ font-size: 1.4rem; margin: 48px 0 16px; }}
  .aulas {{ display: grid; gap: 20px; grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr)); }}
  .aula {{
    background: var(--card); border: 1px solid var(--line); border-radius: 12px;
    overflow: hidden; display: flex; flex-direction: column;
  }}
  .aula .midia {{ aspect-ratio: 16 / 10; background: #000; overflow: hidden; }}
  .aula .midia pre {{
    margin: 0; height: 100%; padding: 20px; overflow: hidden; color: #c9d1d9;
    font: 15px/1.7 ui-monospace, SFMono-Regular, Menlo, monospace;
    background: linear-gradient(160deg, #1b2230, #0d1117);
  }}
  .aula .midia pre b {{ color: var(--accent); font-weight: 600; }}
  .aula .corpo {{ padding: 18px 20px 22px; display: flex; flex-direction: column; flex: 1; }}
  .num {{ color: var(--accent); font-weight: 700; font-size: .85rem; letter-spacing: .08em; }}
  .aula h3 {{ margin: 4px 0 8px; font-size: 1.2rem; }}
  .aula p {{ margin: 0 0 16px; color: var(--muted); font-size: .97rem; flex: 1; }}
  .meta {{ font-size: .85rem; color: var(--muted); margin-bottom: 14px; }}
  .botao {{
    display: inline-flex; align-items: center; gap: 10px; align-self: flex-start;
    background: var(--accent); color: #111; font-weight: 700; text-decoration: none;
    padding: 10px 16px; border-radius: 8px;
  }}
  .botao:hover {{ filter: brightness(1.08); }}
  .passos {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 8px 24px; }}
  .passos li {{ margin: 10px 0; }}
  code {{ background: #262c36; padding: 1px 6px; border-radius: 4px; font-size: .92em; }}
  a {{ color: var(--link); }}
  .exercicios {{ display: grid; gap: 16px; grid-template-columns: repeat(auto-fill, minmax(min(100%, 300px), 1fr)); }}
  .ex {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 18px 20px; display: flex; flex-direction: column; }}
  .ex-topo {{ display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }}
  .ex-num {{ background: var(--accent); color: #111; font-weight: 800; border-radius: 6px; padding: 2px 8px; font-size: .9rem; }}
  .ex-aula {{ color: var(--muted); font-size: .8rem; letter-spacing: .02em; }}
  .ex h3 {{ margin: 4px 0 6px; font-size: 1.08rem; }}
  .ex p {{ margin: 0 0 14px; color: var(--muted); font-size: .92rem; flex: 1; }}
  .ex .botao {{ padding: 8px 14px; font-size: .92rem; }}
  footer {{ margin-top: 56px; color: var(--muted); font-size: .9rem; border-top: 1px solid var(--line); padding-top: 20px; }}
</style>
</head>
<body>
<main>
  <header>
    <h1>Transfer learning na prática: <span>o peso que já veio</span></h1>
    <p class="lead">Uma rede vazia precisa de milhares de fotos para aprender o que é uma borda. Uma ResNet que já viu o ImageNet chega com esse ofício pronto. O que muda é a cabeça, o pedaço que diz o rótulo do seu caso.</p>
    <p class="lead">São quatro notebooks curtos. Cada um abre no Google Colab, baixa o peso uma vez e roda em CPU. O desenho é círculo e quadrado, para a conta caber em dois minutos. Foto de rua, de exame ou de fábrica pede o mesmo desenho e bem mais cuidado na prova.</p>
    <p class="lead">São 24 exercícios, seis por aula, e cada um tem um notebook próprio. Você escreve a resposta, roda a célula e o notebook diz na hora se acertou. <a href="#exercicios">Ver os 24 exercícios</a>.</p>
  </header>

  <div class="hero">
    <pre>for p in base.parameters():
    p.requires_grad = <b>False</b>

base.fc = Linear(512, <b>suas classes</b>)</pre>
  </div>

  <h2>As aulas</h2>
  <div class="aulas">

{aulas_html()}

  </div>

  <h2 id="exercicios">Os 24 exercícios</h2>
  <p class="lead">Cada exercício tem um notebook só dele. Ele prepara o modelo da aula, deixa uma célula para você tentar e mostra uma solução conferida. É só clicar em <strong>Abrir no Colab</strong>.</p>
  <div class="exercicios">
{cards}
  </div>

  <h2>Como rodar</h2>
  <ol class="passos">
    <li>Clique em <strong>Abrir no Colab</strong> e entre com sua conta Google.</li>
    <li>Menu <strong>Ambiente de execução → Executar tudo</strong>. CPU basta. O peso da ResNet baixa na primeira execução.</li>
    <li>Se o Colab avisar que o notebook não é do Google, confirme em <strong>Executar mesmo assim</strong>.</li>
    <li>Nos exercícios, troque <code>resposta = None</code> pela sua resposta e rode a célula. A solução conferida vem na célula de baixo.</li>
    <li>Para guardar suas alterações, use <strong>Arquivo → Salvar uma cópia no Drive</strong>.</li>
  </ol>

  <footer>
    O código está em <a href="{REPO}">github.com/naubergois/curso-transfer-colab</a>.
    Os pesos vêm do ImageNet, pelo torchvision.
    A detecção com caixas está em <a href="https://naubergois.com/curso-yolo/">YOLO na prática</a>.
    <br>Nauber Gois · <a href="https://naubergois.com">naubergois.com</a>
  </footer>
</main>
</body>
</html>
'''


def readme():
    linhas = [
        "# Transfer learning no Colab",
        "",
        "Quatro notebooks curtos. A ResNet-18 chega com os pesos do ImageNet, a cabeça nova aprende o rótulo, a base fica parada, e a prova é um conjunto que o treino não viu.",
        "",
        "| Aula | Abrir |",
        "|---|---|",
    ]
    titulos = [
        "1. O peso que viajou",
        "2. Congelar e soltar",
        "3. Poucas fotos",
        "4. Medir no seu dado",
    ]
    for titulo, (_num, _t, _texto, _meta, caminho, _codigo) in zip(titulos, AULAS_PAGINA):
        url = colab_url(caminho)
        linhas.append(
            f"| {titulo} | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url}) |"
        )
    linhas += [
        "",
        "No Colab, use **Ambiente de execução → Executar tudo**. CPU basta. O arquivo de pesos baixa uma vez.",
        "",
        "## Os 24 exercícios, um notebook cada",
        "",
        "Cada exercício também existe como notebook próprio: prepara o modelo, deixa uma célula para você tentar e mostra uma solução conferida.",
        "",
        "| Aula | Exercício | Abrir |",
        "|---|---|---|",
    ]
    for _nb, aula, n, ex, arquivo in lista_exercicios():
        url = colab_url("exercicios/" + arquivo)
        linhas.append(f"| {aula} | {n}. {ex['titulo']} | [Colab]({url}) |")
    linhas += [
        "",
        "Os exercícios ficam em `exercicios.py`. Os notebooks avulsos são gerados por `build_exercicios.py`.",
        "",
        "```bash",
        "python build_notebooks.py",
        "python build_exercicios.py",
        "python build_pagina.py",
        "```",
        "",
        "A página fica em [naubergois.com/curso-transfer](https://naubergois.com/curso-transfer/).",
        "",
    ]
    return "\n".join(linhas)


def main():
    destino = PORTAL / "index.html"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(pagina(), encoding="utf-8")
    (RAIZ / "README.md").write_text(readme(), encoding="utf-8")
    print("ok", destino)


if __name__ == "__main__":
    main()
