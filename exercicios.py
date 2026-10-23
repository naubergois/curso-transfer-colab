"""Exercícios com correção automática, seis por aula."""

CONFERIR = '''
_acertos = {}


def conferir(n, resposta):
    if resposta is None:
        print(f"Exercício {n}: ainda sem resposta.")
        return
    try:
        certo = bool(VERIFICAR[n](resposta))
    except Exception as erro:
        certo = False
        print(f"  (a verificação deu erro: {erro})")
    _acertos[n] = certo
    print(f"Exercício {n}: " + ("✅ certo!" if certo else "❌ ainda não. Releia o enunciado e tente de novo."))


def placar():
    print(f"{sum(_acertos.values())} de {len(VERIFICAR)} exercícios certos.")
'''


def id_exercicio(notebook, n):
    return f"exercicio-{notebook[:2]}-{n}"


def celulas_exercicios(aula, solucoes=False):
    celulas = [
        ("md", """
## Exercícios

Seis contas em cima do que já está na memória. Escreva a resposta na variável `resposta` e rode a célula: `conferir` diz na hora se fechou. Se travar, abra **Ver solução**.
"""),
        ("form", "# @title Preparação dos exercícios (rode, não precisa ler)\n" + aula["preparo"].strip("\n") + "\n" + CONFERIR),
    ]
    for n, ex in enumerate(aula["lista"], start=1):
        solucao = ex["solucao"].strip("\n")
        celulas.append(("md", f"""
### Exercício {n}. {ex["titulo"]}

{ex["enunciado"].strip()}

<details>
<summary>Ver solução</summary>

```python
{solucao}
```

</details>
"""))
        corpo = solucao if solucoes else f"# Dica: {ex['dica']}\nresposta = None"
        celulas.append(("code", f"{corpo}\n\nconferir({n}, resposta)"))
    celulas.append(("code", "placar()"))
    return celulas


AULA1 = {
    "preparo": '''
VERIFICAR = {
    1: lambda r: int(r) == contar(rede),
    2: lambda r: int(r) == saida_antiga,
    3: lambda r: int(r) == entrada,
    4: lambda r: int(r) == contar(modelo, so_treinavel=True),
    5: lambda r: int(r) == contar(modelo) - contar(modelo, so_treinavel=True),
    6: lambda r: bool(r) is False,
}
''',
    "lista": [
        {
            "titulo": "Quantos pesos vieram",
            "enunciado": "Quantos parâmetros a `rede` pré-treinada tem, antes de trocar a cabeça?",
            "dica": "contar(rede) soma o numel de cada parâmetro.",
            "solucao": "resposta = contar(rede)",
        },
        {
            "titulo": "A cabeça antiga",
            "enunciado": "Quantas classes a cabeça antiga sabia responder? É o `out_features` da `rede`.",
            "dica": "rede.fc.out_features",
            "solucao": "resposta = rede.fc.out_features",
        },
        {
            "titulo": "O tamanho do vetor",
            "enunciado": "Quantos números a base entrega para a cabeça? É o `in_features`.",
            "dica": "A variável entrada já guardou esse número. rede.fc.in_features também.",
            "solucao": "resposta = rede.fc.in_features",
        },
        {
            "titulo": "O que ainda treina",
            "enunciado": "Depois de congelar a base e trocar a cabeça, quantos parâmetros de `modelo` ainda podem mudar?",
            "dica": "contar(modelo, so_treinavel=True) ignora quem está com requires_grad False.",
            "solucao": "resposta = contar(modelo, so_treinavel=True)",
        },
        {
            "titulo": "O que ficou parado",
            "enunciado": "Quantos parâmetros de `modelo` ficaram congelados?",
            "dica": "O total menos os que ainda treinam.",
            "solucao": "resposta = contar(modelo) - contar(modelo, so_treinavel=True)",
        },
        {
            "titulo": "A primeira camada",
            "enunciado": "O peso `modelo.conv1.weight` ainda pode mudar? Responda `True` ou `False`.",
            "dica": "requires_grad diz se o otimizador tem permissão de mexer.",
            "solucao": "resposta = modelo.conv1.weight.requires_grad",
        },
    ],
}

AULA2 = {
    "preparo": '''
VERIFICAR = {
    1: lambda r: bool(r) is False,
    2: lambda r: bool(r) is True,
    3: lambda r: int(r) == len(otimizador.param_groups),
    4: lambda r: abs(float(r) - otimizador.param_groups[0]["lr"]) < 1e-12,
    5: lambda r: abs(float(r) - otimizador.param_groups[1]["lr"]) < 1e-12,
    6: lambda r: int(r) == contar(modelo.layer4),
}
''',
    "lista": [
        {
            "titulo": "A porta de entrada",
            "enunciado": "`conv1` é a primeira camada. Ela voltou a treinar? Responda `True` ou `False`.",
            "dica": "modelo.conv1.weight.requires_grad",
            "solucao": "resposta = modelo.conv1.weight.requires_grad",
        },
        {
            "titulo": "O bloco que soltou",
            "enunciado": "Todos os parâmetros de `layer4` estão livres para treinar? Responda `True` ou `False`.",
            "dica": "all(p.requires_grad for p in modelo.layer4.parameters())",
            "solucao": "resposta = all(p.requires_grad for p in modelo.layer4.parameters())",
        },
        {
            "titulo": "Dois grupos",
            "enunciado": "Quantos grupos de taxa de aprendizado o `otimizador` tem?",
            "dica": "otimizador.param_groups é a lista. Cada grupo tem o seu lr.",
            "solucao": "resposta = len(otimizador.param_groups)",
        },
        {
            "titulo": "O passo da base",
            "enunciado": "Qual é a taxa de aprendizado do primeiro grupo, o da base que soltou?",
            "dica": "otimizador.param_groups[0]['lr']",
            "solucao": "resposta = otimizador.param_groups[0]['lr']",
        },
        {
            "titulo": "O passo da cabeça",
            "enunciado": "Qual é a taxa de aprendizado do segundo grupo, o da cabeça nova?",
            "dica": "O índice 1 é a cabeça.",
            "solucao": "resposta = otimizador.param_groups[1]['lr']",
        },
        {
            "titulo": "Pesos do bloco",
            "enunciado": "Quantos parâmetros tem o `layer4`, livres e parados somados?",
            "dica": "contar(modelo.layer4)",
            "solucao": "resposta = contar(modelo.layer4)",
        },
    ],
}

AULA3 = {
    "preparo": '''
VERIFICAR = {
    1: lambda r: int(r) == len(y_treino),
    2: lambda r: int(r) == int(y_treino.unique().numel()),
    3: lambda r: int(r) == epocas,
    4: lambda r: float(r) < 1e-6,
    5: lambda r: float(r) > 1e-4,
    6: lambda r: abs(float(r) - acuracia(modelo, x_teste, y_teste)) < 1e-5,
}
''',
    "lista": [
        {
            "titulo": "Quantas fotos de treino",
            "enunciado": "Quantas imagens tem o conjunto de treino, `y_treino`?",
            "dica": "len(y_treino)",
            "solucao": "resposta = len(y_treino)",
        },
        {
            "titulo": "Quantas classes",
            "enunciado": "Quantos rótulos diferentes aparecem em `y_treino`? Círculo e quadrado são dois.",
            "dica": "y_treino.unique() devolve os valores distintos.",
            "solucao": "resposta = int(y_treino.unique().numel())",
        },
        {
            "titulo": "Quantas épocas",
            "enunciado": "Quantas épocas o treino deu na cabeça?",
            "dica": "A variável epocas guarda isso.",
            "solucao": "resposta = epocas",
        },
        {
            "titulo": "A base não andou",
            "enunciado": "Qual foi a maior mudança, em valor absoluto, no peso `conv1` depois do treino? Um número perto de zero quer dizer que a base ficou parada.",
            "dica": "(modelo.conv1.weight.detach() - antes_conv).abs().max().item()",
            "solucao": "resposta = (modelo.conv1.weight.detach() - antes_conv).abs().max().item()",
        },
        {
            "titulo": "A cabeça andou",
            "enunciado": "Qual foi a maior mudança no peso da cabeça, `modelo.fc.weight`, comparado com `antes_fc`?",
            "dica": "A mesma conta da base, agora em modelo.fc.weight e antes_fc.",
            "solucao": "resposta = (modelo.fc.weight.detach() - antes_fc).abs().max().item()",
        },
        {
            "titulo": "Acerto no teste",
            "enunciado": "Qual a fração de acerto em `x_teste` e `y_teste`? Use `acuracia`. O teste não entrou no treino.",
            "dica": "acuracia(modelo, x_teste, y_teste) devolve um número entre 0 e 1.",
            "solucao": "resposta = acuracia(modelo, x_teste, y_teste)",
        },
    ],
}

AULA4 = {
    "preparo": '''
VERIFICAR = {
    1: lambda r: int(r) == A.shape[0],
    2: lambda r: int(r) == A.numel() + B.numel(),
    3: lambda r: int(r) == W.numel(),
    4: lambda r: int(r) == len(y_prova),
    5: lambda r: abs(float(r) - acuracia_linear(peso, vies, feats_prova, y_prova)) < 1e-5,
    6: lambda r: abs(float(r) - acuracia_linear(peso, vies, feats_longe, y_longe)) < 1e-5,
}
''',
    "lista": [
        {
            "titulo": "O posto do par",
            "enunciado": "Qual é o posto do par pequeno? O posto é a dimensão do meio, a primeira dimensão de `A`.",
            "dica": "A.shape[0]",
            "solucao": "resposta = A.shape[0]",
        },
        {
            "titulo": "Pesos do par",
            "enunciado": "Quantos números treináveis o par `A` e `B` tem, somados?",
            "dica": "A.numel() + B.numel()",
            "solucao": "resposta = A.numel() + B.numel()",
        },
        {
            "titulo": "Pesos da matriz cheia",
            "enunciado": "Quantos números tem a matriz `W`, a que a gente não mexe?",
            "dica": "W.numel()",
            "solucao": "resposta = W.numel()",
        },
        {
            "titulo": "Tamanho da prova",
            "enunciado": "Quantas fotos tem a prova, `y_prova`? Essa prova não entrou no ajuste da cabeça.",
            "dica": "len(y_prova)",
            "solucao": "resposta = len(y_prova)",
        },
        {
            "titulo": "Acerto no seu dado",
            "enunciado": "Qual a fração de acerto da cabeça linear em `feats_prova`? Use `acuracia_linear`.",
            "dica": "acuracia_linear(peso, vies, feats_prova, y_prova)",
            "solucao": "resposta = acuracia_linear(peso, vies, feats_prova, y_prova)",
        },
        {
            "titulo": "Acerto no dado longe",
            "enunciado": "A mesma cabeça, agora em `feats_longe`, onde a cor dos desenhos foi trocada. Qual a fração de acerto?",
            "dica": "acuracia_linear(peso, vies, feats_longe, y_longe)",
            "solucao": "resposta = acuracia_linear(peso, vies, feats_longe, y_longe)",
        },
    ],
}

AULA5 = {
    "preparo": '''
import tensorflow as tf

VERIFICAR = {
    1: lambda r: int(r) == sum(int(tf.reduce_prod(tf.shape(p)).numpy()) for p in modelo.trainable_weights),
}
''',
    "lista": [
        {
            "titulo": "Pesos que ainda treinam",
            "enunciado": "A base do MobileNetV2 ficou congelada. Quantos parâmetros de `modelo` ainda podem mudar? É a cabeça nova, a que separa normal de pneumonia.",
            "dica": "modelo.trainable_weights lista só o que o Adam pode mexer. Some o tamanho de cada um.",
            "solucao": "resposta = sum(int(tf.reduce_prod(tf.shape(p)).numpy()) for p in modelo.trainable_weights)",
        },
    ],
}

EXERCICIOS = {
    "01-o-peso-que-viajou.ipynb": AULA1,
    "02-congelar-e-soltar.ipynb": AULA2,
    "03-poucas-fotos.ipynb": AULA3,
    "04-medir-no-seu-dado.ipynb": AULA4,
    "05-pneumonia-keras.ipynb": AULA5,
}
