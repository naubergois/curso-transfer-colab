"""Gera os notebooks do curso em notebooks/*.ipynb.

Com SOLUCOES=1, preenche os exercícios com as soluções (para testar a correção).
Com OUT=<pasta>, grava em outra pasta.
"""

import os
import re
from pathlib import Path

import nbformat as nbf

from blocos import CONTAR, SETUP_1, SETUP_2, SETUP_3, SETUP_4, SETUP_5
from comum import INSTALL, badge, novo_notebook
from exercicios import EXERCICIOS, celulas_exercicios, id_exercicio

OUT = Path(os.environ.get("OUT", Path(__file__).parent / "notebooks"))
SOLUCOES = os.environ.get("SOLUCOES") == "1"


def notebook(nome, celulas):
    nb = novo_notebook()
    if nome in EXERCICIOS:
        celulas = celulas[:-1] + celulas_exercicios(EXERCICIOS[nome], SOLUCOES) + celulas[-1:]
    out = []
    for tipo, texto in celulas:
        texto = texto.strip("\n")
        texto = texto.replace("{BADGE}", badge(f"notebooks/{nome}"))
        if tipo == "md":
            cel = nbf.v4.new_markdown_cell(texto)
            ex = re.match(r"### Exercício (\d+)\.", texto)
            if ex:
                cel.metadata["id"] = id_exercicio(nome, int(ex.group(1)))
            out.append(cel)
        else:
            cel = nbf.v4.new_code_cell(texto)
            if tipo == "form":
                cel.metadata["cellView"] = "form"
            out.append(cel)
    nb.cells = out
    OUT.mkdir(exist_ok=True)
    nbf.write(nb, OUT / nome)
    print("ok", nome)


notebook("01-o-peso-que-viajou.ipynb", [
    ("md", """
{BADGE}

# 1. O peso que viajou

Uma rede nova nasce sem opinião. Os pesos são o que ela já viu. Transfer learning pega esses pesos, treinados em milhões de fotos do ImageNet, e leva para um caso com poucas fotos suas.

A rede usada aqui é a ResNet-18. A última camada, a cabeça, responde 1000 classes: cachorro, carro, cadeira. A base, tudo que vem antes, descreve a foto num vetor. Troca a cabeça. A base fica.

> No Colab: **Ambiente de execução → Executar tudo**. CPU basta. O arquivo de pesos baixa uma vez.
"""),
    ("code", INSTALL),
    ("md", """
## A base chega pronta, a cabeça sai

`resnet18(weights=...)` baixa o ofício. `rede` fica intacta, para a gente contar o que veio. `modelo` é a cópia de trabalho: a base inteira fica com `requires_grad = False`, e a cabeça vira uma camada linear de duas saídas.

Duas saídas são o caso novo. Círculo e quadrado, ou as duas classes que forem suas. Isso ainda não treina nada. Só mostra o que viajou e o que ficou livre.
"""),
    ("code", CONTAR + "\n" + SETUP_1),
    ("md", """
## O que fica

Mil classes emprestadas não são o seu problema. O número que importa é o dos pesos que ainda podem mudar: só a cabeça nova. O viés do ImageNet viaja junto com a base. Gente, cão e carro estavam naquele saco. O seu caso pode não estar.

Os seis exercícios abaixo usam estes mesmos nomes: `rede`, `modelo`, `entrada`, `saida_antiga` e `contar`.
"""),
])

notebook("02-congelar-e-soltar.ipynb", [
    ("md", """
{BADGE}

# 2. Congelar e soltar

Com poucas fotos, deixar a base inteira se mexer apaga o ofício. O desenho seguro é o contrário: a base fica quieta, a cabeça aprende o rótulo, e só depois o último bloco da base volta, com um passo curto.

Na ResNet esse último bloco se chama `layer4`. A primeira camada, `conv1`, continua travada. Ela já sabe achar borda e mancha. Não precisa reaprender isso com trinta fotos.
"""),
    ("code", INSTALL),
    ("md", """
## Duas velocidades no mesmo passo

O otimizador aceita grupos. Cada grupo tem a sua taxa. A base que soltou anda a `1e-4`. A cabeça nova anda a `1e-3`, dez vezes mais. Passo grande na base desaprende. Passo grande na cabeça, que nasceu agora, é o que faz ela sair do zero.
"""),
    ("code", CONTAR + "\n" + SETUP_2),
    ("md", """
## O que fica

Soltar não é soltar tudo. `conv1` segue parada, `layer4` volta, e as duas taxas convivem no mesmo `Adam`. Isso ainda não é o treino nas suas fotos. É o freio e o acelerador, prontos antes de andar.

Os exercícios usam `modelo`, `otimizador` e `contar`.
"""),
])

notebook("03-poucas-fotos.ipynb", [
    ("md", """
{BADGE}

# 3. Poucas fotos

A base emprestada já viu milhões de imagens. Com um punhado de desenhos seus, a cabeça aprende o rótulo. Uma rede vazia, do mesmo tamanho, pediria bem mais exemplo para chegar no mesmo lugar.

O desenho daqui é simples de propósito: círculo vermelho ou quadrado azul, 32 fotos para treinar e 16 que ela não viu. Duas épocas. A base fica em modo de avaliação, para a estatística que ela já calculou não ser reescrita.
"""),
    ("code", INSTALL),
    ("md", """
## A cabeça anda, a base não

Antes do primeiro passo, o peso de `conv1` e o peso da cabeça são copiados. Depois do treino, a diferença máxima diz quem mexeu. Na base, a conta tem que dar zero. Na cabeça, tem que sair do lugar.

O acerto é medido nas 16 fotos de teste. Elas não entraram no `Adam`.
"""),
    ("code", CONTAR + "\n" + SETUP_3),
    ("md", """
## O que fica

Trinta e duas fotos de círculo e quadrado mostram o caminho. Isso ainda não é um modelo para foto de rua, de exame ou de fábrica. O que dá para levar é o teste: separou um conjunto, treinou sem ele, e leu o acerto nele.

Os exercícios usam `y_treino`, `epocas`, `antes_conv`, `antes_fc`, `acuracia`, `x_teste` e `y_teste`.
"""),
])

notebook("04-medir-no-seu-dado.ipynb", [
    ("md", """
{BADGE}

# 4. Medir no seu dado

Acurácia no conjunto emprestado não prova o seu caso. A base herda o olhar de quem montou o ImageNet. Erro de lá pode aparecer aqui. A prova é um grupo de fotos suas, separado do ajuste.

Quando a rede é grande demais para treinar por inteiro, um par magro fica ao lado do peso original. Duas matrizes finas, `A` e `B`, com um posto pequeno no meio. O peso grande não se mexe. Num modelo de linguagem esse par se chama LoRA. Aqui a matriz é de brinquedo, 512 por 512, para a conta caber na tela. O posto 4 é escolha deste exemplo.
"""),
    ("code", INSTALL),
    ("md", """
## O par, e a prova com a cor trocada

`W` é a matriz que ninguém treina. `A` tem forma `(4, 512)` e `B` tem forma `(512, 4)`. O produto `B @ A` tem o mesmo formato de `W`, e a soma dos dois pesos do par é bem menor que `W`.

Em seguida a base vira um medidor: a cabeça some, fica o vetor de 512 números, e uma camada linear aprende círculo contra quadrado em cima desse vetor. A prova repete a conta. O conjunto longe é o mesmo desenho com a cor trocada. Se os dois acertos ficarem perto, o deslocamento foi pequeno. Um raio-x de verdade fica mais longe do que inverter o canal.
"""),
    ("code", CONTAR + "\n" + SETUP_4),
    ("md", """
## O que fica

Peso emprestado não vem neutro. O número que decide é o acerto no seu conjunto, e vale repetir a medida quando o dado muda de jeito. O par pequeno é um jeito de acrescentar tarefa sem reescrever a matriz. Isso ainda não treina um modelo de linguagem.

Os exercícios usam `A`, `B`, `W`, `y_prova`, `acuracia_linear`, `feats_prova` e `feats_longe`.
"""),
])

notebook("05-pneumonia-keras.ipynb", [
    ("md", """
{BADGE}

# 5. Pneumonia com Keras

O raio-x de tórax chega em duas pastas: normal e pneumonia. São radiografias de um conjunto público, rotuladas por quem examina o filme. A rede não nasce vazia. O MobileNetV2 já viu milhões de fotos comuns. A base fica parada. A cabeça nova, feita no Keras, só aprende a escolher entre essas duas pastas.

Duas épocas num punhado de exames mostram o caminho. Isso ainda não é um laudo. O peso veio de foto de objeto, não de hospital, e a prova daqui é pequena.
"""),
    ("code", "!pip install -q tensorflow"),
    ("md", """
## A base emprestada, a cabeça do exame

O zip é a parte de validação do conjunto Chest X-Ray (Kermany e colegas, CC BY 4.0), espelhado no Hugging Face. De cada pasta saem 24 imagens para treinar e 8 que ficam de fora.

`include_top=False` tira a cabeça das mil classes do ImageNet. `trainable = False` trava o resto. A camada `Dense` de duas saídas é o único pedaço que o `Adam` mexe.
"""),
    ("code", SETUP_5),
    ("md", """
## O que fica

A base descreve a imagem. A cabeça diz normal ou pneumonia. O número que o exercício pede é o dos pesos que ainda podem mudar. O acerto impresso acima foi medido nas radiografias que não entraram no ajuste. Ele não autoriza decisão clínica.

O exercício usa `modelo` e `tf`.
"""),
])
