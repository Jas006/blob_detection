# blob_detection
Détecter et catégoriser les brins d'ARN d'une image

Listes des modules à importer:

Modules à importer

import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import blob_log
from skimage.color import rgb2gray
import czifile
from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import white_tophat as hat
from skimage.morphology import reconstruction
from scipy.optimize import leastsq
import xlsxwriter

import tkinter as tk
from tkinter import filedialog
import webbrowser

Instructions pour exécuter le code:

Jusqu'à maintenant, seul le code peremettant l'analyse des images czi a été complété.
Votre image czi peut avoir n'importe quel nombre de stacks et de channels.
Vous pouvez choisir les channels que vous voulez analyser.
Dans la deuxième fenêtre qui s'ouvre, les paramètres à suggérés sont écrit entre parenthèses.
Lorsque vous ne savez pas ce que le paramètre veut dire, vous pouvez cliquer sur l'énoncé; c'est un
lien hypertexte qui vous mènera à une page web vous expliquant sa définition.
Quand vous cliquez sur le bouton poursuivre analyse, une fenêtre apparaîtra et vous expliquera la procédure à suivre.
Pour une image (2048 x 2048),le temps d'exécution avoisinne les 15 minutes.

Dans le dossier d'analyse, vous retrouverez les images analysées et les caractéristiques de toutes les détections.
Vous devez exécuter le fichier Interface_graphique.py pour débuter le GUI
