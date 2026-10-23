"""Partes compartilhadas pelos geradores de notebooks."""

import nbformat as nbf

REPO = "naubergois/curso-transfer-colab"

INSTALL = "!pip install -q torch torchvision"


def colab_url(caminho):
    return f"https://colab.research.google.com/github/{REPO}/blob/main/{caminho}"


def badge(caminho):
    return (f'<a href="{colab_url(caminho)}" target="_blank">'
            '<img src="https://colab.research.google.com/assets/colab-badge.svg" '
            'alt="Abrir no Colab"/></a>')


def novo_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "colab": {"provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    }
    return nb
