"""Trechos de código que a aula e o exercício compartilham.

O exercício roda sozinho, então o setup de cada aula deixa os mesmos nomes
que a aula usa: rede, modelo, contar, e daí em diante.
"""

CONTAR = '''
def contar(modelo, so_treinavel=False):
    total = 0
    for p in modelo.parameters():
        if so_treinavel and not p.requires_grad:
            continue
        total += p.numel()
    return total
'''

FORMAS = '''
def lote_formas(n, semente, trocar_cor=False):
    """Desenha círculos e quadrados. Classe 0 é círculo, classe 1 é quadrado."""
    g = torch.Generator().manual_seed(semente)
    y = torch.randint(0, 2, (n,), generator=g)
    x = torch.full((n, 3, 64, 64), 0.08)
    yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    circulo = (yy - 32) ** 2 + (xx - 32) ** 2 <= 16 ** 2
    for i in range(n):
        cor = torch.tensor([0.90, 0.20, 0.15] if int(y[i]) == 0 else [0.15, 0.35, 0.92])
        if trocar_cor:
            cor = cor.flip(0)
        if int(y[i]) == 0:
            x[i, :, circulo] = cor.view(3, 1)
        else:
            x[i, :, 16:48, 16:48] = cor.view(3, 1, 1)
    return x, y


def normalizar(x):
    media = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    desvio = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - media) / desvio
'''

SETUP_1 = '''
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

torch.manual_seed(7)

rede = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
rede.eval()
entrada = rede.fc.in_features
saida_antiga = rede.fc.out_features

modelo = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
modelo.eval()
for p in modelo.parameters():
    p.requires_grad = False
modelo.fc = nn.Linear(entrada, 2)

print(f"pesos que vieram: {contar(rede)}")
print(f"cabeça antiga: {entrada} -> {saida_antiga}")
print(f"cabeça nova: {modelo.fc.in_features} -> {modelo.fc.out_features}")
print(f"ainda treinam: {contar(modelo, so_treinavel=True)}")
print(f"parados: {contar(modelo) - contar(modelo, so_treinavel=True)}")
print(f"conv1 pode mudar? {modelo.conv1.weight.requires_grad}")
'''

SETUP_2 = '''
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

torch.manual_seed(7)

modelo = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
modelo.eval()
entrada = modelo.fc.in_features
for p in modelo.parameters():
    p.requires_grad = False
modelo.fc = nn.Linear(entrada, 2)
for p in modelo.layer4.parameters():
    p.requires_grad = True

grupo_base = [p for p in modelo.layer4.parameters() if p.requires_grad]
grupo_cabeca = list(modelo.fc.parameters())
otimizador = torch.optim.Adam([
    {"params": grupo_base, "lr": 1e-4},
    {"params": grupo_cabeca, "lr": 1e-3},
])

print(f"conv1 pode mudar? {modelo.conv1.weight.requires_grad}")
print(f"layer4 pode mudar? {all(p.requires_grad for p in modelo.layer4.parameters())}")
print(f"grupos de taxa: {len(otimizador.param_groups)}")
print(f"passo da base: {otimizador.param_groups[0]['lr']}")
print(f"passo da cabeça: {otimizador.param_groups[1]['lr']}")
print(f"pesos do layer4: {contar(modelo.layer4)}")
'''

SETUP_3 = FORMAS + '''
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import ResNet18_Weights, resnet18

torch.manual_seed(7)
epocas = 2
x_treino, y_treino = lote_formas(32, semente=11)
x_teste, y_teste = lote_formas(16, semente=29)
x_treino, x_teste = normalizar(x_treino), normalizar(x_teste)

modelo = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
for p in modelo.parameters():
    p.requires_grad = False
modelo.fc = nn.Linear(modelo.fc.in_features, 2)
modelo.eval()
antes_conv = modelo.conv1.weight.detach().clone()
antes_fc = modelo.fc.weight.detach().clone()

criterio = nn.CrossEntropyLoss()
opt = torch.optim.Adam(modelo.fc.parameters(), lr=1e-2)
loader = DataLoader(TensorDataset(x_treino, y_treino), batch_size=8, shuffle=False)
for _ in range(epocas):
    for xb, yb in loader:
        opt.zero_grad()
        perda = criterio(modelo(xb), yb)
        perda.backward()
        opt.step()

mudou_base = (modelo.conv1.weight.detach() - antes_conv).abs().max().item()
mudou_cabeca = (modelo.fc.weight.detach() - antes_fc).abs().max().item()


def acuracia(modelo, x, y):
    modelo.eval()
    with torch.no_grad():
        pred = modelo(x).argmax(1)
    return float((pred == y).float().mean())


acerto = acuracia(modelo, x_teste, y_teste)
print(f"fotos de treino: {len(y_treino)} | classes: {int(y_treino.unique().numel())} | épocas: {epocas}")
print(f"a base andou: {mudou_base:.3e} | a cabeça andou: {mudou_cabeca:.3e}")
print(f"acerto no teste: {acerto:.0%}")
'''

SETUP_4 = FORMAS + '''
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

torch.manual_seed(7)
posto = 4
entrada = 512
A = torch.zeros(posto, entrada)
B = torch.zeros(entrada, posto)
W = torch.zeros(entrada, entrada)
params_par = A.numel() + B.numel()
params_cheia = W.numel()

x_treino, y_treino = lote_formas(32, semente=11)
x_prova, y_prova = lote_formas(16, semente=29)
x_longe, y_longe = lote_formas(16, semente=29, trocar_cor=True)
x_treino, x_prova, x_longe = normalizar(x_treino), normalizar(x_prova), normalizar(x_longe)

base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
base.fc = nn.Identity()
base.eval()


def extrair(x):
    with torch.no_grad():
        return base(x)


feats_treino = extrair(x_treino)
feats_prova = extrair(x_prova)
feats_longe = extrair(x_longe)

peso = torch.zeros(2, feats_treino.shape[1], requires_grad=True)
vies = torch.zeros(2, requires_grad=True)
opt = torch.optim.Adam([peso, vies], lr=1e-2)
for _ in range(40):
    opt.zero_grad()
    perda = nn.functional.cross_entropy(feats_treino @ peso.T + vies, y_treino)
    perda.backward()
    opt.step()
peso, vies = peso.detach(), vies.detach()


def acuracia_linear(peso, vies, feats, y):
    with torch.no_grad():
        pred = (feats @ peso.T + vies).argmax(1)
    return float((pred == y).float().mean())


acerto_seu = acuracia_linear(peso, vies, feats_prova, y_prova)
acerto_longe = acuracia_linear(peso, vies, feats_longe, y_longe)
print(f"posto {posto} | par {params_par} pesos | matriz cheia {params_cheia} pesos")
print(f"prova: {len(y_prova)} fotos | acerto no seu dado {acerto_seu:.0%} | acerto com a cor trocada {acerto_longe:.0%}")
print(f"o par cabe na matriz? {tuple((B @ A).shape) == tuple(W.shape)}")
'''

SETUP_5 = '''
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

URL_RAIOX = (
    "https://huggingface.co/datasets/keremberke/chest-xray-classification"
    "/resolve/main/data/valid.zip"
)
arquivo = Path("valid.zip")
if not arquivo.exists():
    urllib.request.urlretrieve(URL_RAIOX, arquivo)
if not Path("raiox/NORMAL").exists():
    zipfile.ZipFile(arquivo).extractall("raiox")

rng = random.Random(7)

def partir(classe, n_treino, n_prova):
    fotos = sorted((Path("raiox") / classe).glob("*.jpg"))
    rng.shuffle(fotos)
    return fotos[:n_treino], fotos[n_treino:n_treino + n_prova]

for split in ("treino", "prova"):
    shutil.rmtree(split, ignore_errors=True)
for classe in ("NORMAL", "PNEUMONIA"):
    treino_fotos, prova_fotos = partir(classe, 24, 8)
    for split, fotos in (("treino", treino_fotos), ("prova", prova_fotos)):
        destino = Path(split) / classe
        destino.mkdir(parents=True)
        for foto in fotos:
            shutil.copy(foto, destino / foto.name)

tamanho = (128, 128)
treino = tf.keras.utils.image_dataset_from_directory(
    "treino", image_size=tamanho, batch_size=8, shuffle=False,
)
prova = tf.keras.utils.image_dataset_from_directory(
    "prova", image_size=tamanho, batch_size=8, shuffle=False,
)
nomes = treino.class_names

for imagens, rotulos in treino.take(1):
    plt.figure(figsize=(8, 3))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(imagens[i].numpy().astype("uint8"))
        plt.title(nomes[int(rotulos[i])])
        plt.axis("off")
    plt.show()
    break

preparar = tf.keras.applications.mobilenet_v2.preprocess_input
treino_modelo = treino.map(lambda imagem, rotulo: (preparar(imagem), rotulo))
prova_modelo = prova.map(lambda imagem, rotulo: (preparar(imagem), rotulo))

base = tf.keras.applications.MobileNetV2(
    input_shape=tamanho + (3,), include_top=False, weights="imagenet",
)
base.trainable = False
entradas = tf.keras.Input(shape=tamanho + (3,))
x = base(entradas, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
saidas = tf.keras.layers.Dense(len(nomes), activation="softmax")(x)
modelo = tf.keras.Model(entradas, saidas)
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
modelo.fit(treino_modelo, epochs=2, verbose=0)
perda, acerto = modelo.evaluate(prova_modelo, verbose=0)
treinaveis = sum(int(tf.reduce_prod(tf.shape(p)).numpy()) for p in modelo.trainable_weights)
print(f"classes: {nomes}")
print(f"base pode mudar? {base.trainable}")
print(f"pesos que ainda treinam: {treinaveis}")
print(f"acerto na prova: {acerto:.0%}")
'''
