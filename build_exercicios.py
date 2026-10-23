"""Gera um notebook por exercício em exercicios/*.ipynb."""

import os
import re
import unicodedata
from pathlib import Path

import nbformat as nbf

from blocos import CONTAR, SETUP_1, SETUP_2, SETUP_3, SETUP_4, SETUP_5
from comum import INSTALL, badge, colab_url, novo_notebook
from exercicios import CONFERIR, EXERCICIOS

OUT = Path(os.environ.get("OUT", Path(__file__).parent / "exercicios"))

AULAS = {
    "01-o-peso-que-viajou.ipynb": {
        "nome": "O peso que viajou",
        "contexto": "A preparação carrega a ResNet-18 com os pesos do ImageNet, guarda a rede intacta e monta uma cópia com a base congelada e a cabeça nova de duas saídas.",
        "setup": CONTAR + "\n" + SETUP_1,
    },
    "02-congelar-e-soltar.ipynb": {
        "nome": "Congelar e soltar",
        "contexto": "A preparação congela a base, troca a cabeça e solta só o `layer4`. O otimizador fica com duas taxas: um passo curto na base, um passo maior na cabeça.",
        "setup": CONTAR + "\n" + SETUP_2,
    },
    "03-poucas-fotos.ipynb": {
        "nome": "Poucas fotos",
        "contexto": "A preparação desenha 32 fotos de treino e 16 de teste, círculos e quadrados, e treina só a cabeça por duas épocas. A base fica em avaliação.",
        "setup": CONTAR + "\n" + SETUP_3,
    },
    "04-medir-no-seu-dado.ipynb": {
        "nome": "Medir no seu dado",
        "contexto": "A preparação monta o par `A` e `B` ao lado da matriz `W`, extrai o vetor da base e ajusta uma cabeça linear. A prova e o conjunto de cor trocada ficam de fora desse ajuste.",
        "setup": CONTAR + "\n" + SETUP_4,
    },
    "05-pneumonia-keras.ipynb": {
        "nome": "Pneumonia com Keras",
        "install": "!pip install -q tensorflow",
        "contexto": "A preparação baixa radiografias de tórax já rotuladas como normal ou pneumonia, congela o MobileNetV2 que veio do ImageNet e treina só a cabeça por duas épocas.",
        "setup": SETUP_5,
    },
}


def slug(texto):
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", texto.lower()).strip("-")


def lista_exercicios():
    saida = []
    for aula_nb, aula in EXERCICIOS.items():
        k = int(aula_nb[:2])
        for n, ex in enumerate(aula["lista"], start=1):
            saida.append((aula_nb, k, n, ex, f"aula{k}-exercicio{n}-{slug(ex['titulo'])}.ipynb"))
    return saida


def gerar():
    todos = lista_exercicios()
    OUT.mkdir(exist_ok=True)
    for i, (aula_nb, k, n, ex, arquivo) in enumerate(todos):
        info = AULAS[aula_nb]
        proximo = todos[i + 1] if i + 1 < len(todos) else None
        aula_url = colab_url(f"notebooks/{aula_nb}")
        if proximo:
            fecho = (f"Próximo: [Aula {proximo[1]}, exercício {proximo[2]}: {proximo[3]['titulo']}]"
                     f"({colab_url('exercicios/' + proximo[4])}).")
        else:
            fecho = (
                f"Esse foi o último dos {len(todos)}. "
                "O acerto numa radiografia de treino ainda não é um laudo. "
                "A prova continua sendo um conjunto que o ajuste não viu."
            )

        celulas = [
            ("md", f"""{badge('exercicios/' + arquivo)}

# Aula {k}, exercício {n}: {ex['titulo']}

{ex['enunciado'].strip()}

O exercício é da aula [{k}. {info['nome']}]({aula_url}), e este notebook roda sozinho. Use **Ambiente de execução → Executar tudo**: ele prepara o modelo, deixa uma célula para você tentar e, no fim, mostra uma solução já conferida."""),
            ("code", info.get("install", INSTALL)),
            ("md", f"## Preparação\n\n{info['contexto']}"),
            ("code", info["setup"]),
            ("form", "# @title Correção automática (rode, não precisa ler)\n"
                     + EXERCICIOS[aula_nb]["preparo"].strip("\n") + "\n" + CONFERIR),
            ("md", f"## Sua vez\n\n{ex['enunciado'].strip()}\n\n*Dica:* {ex['dica']}\n\n"
                   "Troque o `None` pela sua resposta e rode a célula."),
            ("code", f"resposta = None\n\nconferir({n}, resposta)"),
            ("md", "## Uma solução\n\nCompare com o que você fez. Existem outros caminhos certos: o que vale é a `conferir` aprovar."),
            ("code", f"{ex['solucao'].strip()}\n\nconferir({n}, resposta)"),
            ("md", f"## O que fica\n\n{fecho}"),
        ]

        nb = novo_notebook()
        for tipo, texto in celulas:
            if tipo == "md":
                nb.cells.append(nbf.v4.new_markdown_cell(texto.strip("\n")))
            else:
                cel = nbf.v4.new_code_cell(texto.strip("\n"))
                if tipo == "form":
                    cel.metadata["cellView"] = "form"
                nb.cells.append(cel)
        nbf.write(nb, OUT / arquivo)
    print(f"{len(todos)} notebooks em {OUT}")


if __name__ == "__main__":
    gerar()
