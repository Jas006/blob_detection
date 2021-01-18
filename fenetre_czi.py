from fenetre_accueil import *
from czifile import imread
from tkinter import messagebox as mb
import webbrowser
image=imread(alpha.path)
nombre_de_canaux=len(image[0,0])
nombre_de_zstack=len(image[0,0,0,0])

def callback(url):
    webbrowser.open_new(url)

dossier_text=open("text_file_czi.txt","r")
liste_text=dossier_text.readlines()
dossier_text.close()
gauss=int(liste_text[0])
th=int(liste_text[1])
seuil_pourcent=float(liste_text[2])
maximum_blob=int(liste_text[3])
num_blob=int(liste_text[4])
seuil_blob=float(liste_text[5])
distance_s=int(liste_text[6])

class fenetre_analyse(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.liste=[]
        self.liste2=[]
        self.liste_canal=[]
        self.liste3 = []
        self.liste4 = []
        self.liste_z = []
        self.label_check=tk.Label(text="Séléctionner les canaux à analyser").grid(row=1,column=1)
        for i in range(nombre_de_canaux):
            self.liste.append(tk.IntVar())
        for i in range(nombre_de_canaux):
            self.liste2.append(tk.Checkbutton(text=f"Canal{i + 1}", variable=self.liste[i]))
            self.liste2[i].grid(column=1,row=i+3)
        self.label_check_z = tk.Label(text="Séléctionner les z_stack à analyser").grid(column=44,row=1)
        for i in range(nombre_de_zstack):
            self.liste3.append(tk.IntVar())
        for i in range(nombre_de_zstack):
            self.liste4.append(tk.Checkbutton(text=f"z_stack{i + 1}", variable=self.liste3[i]))
            self.liste4[i].grid(column=44,row=i+3)

        self.label_filtre_gauss = tk.Label(
            text=f"Séléctionner le sigma du filtre gaussien (3) \n Actuellement {gauss}")
        self.label_filtre_gauss.grid(column=1, row=15)
        self.label_filtre_gauss.bind("<Button-1>",
                                     lambda e: callback("https://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm"))
        self.sigma_gauss = tk.Entry()
        self.sigma_gauss.grid(column=1, row=17)
        self.label_wth = tk.Label(text=f"Séléctionner la taille du filtre white top hat(30)\n Actuellement {th}")
        self.label_wth.grid(column=1, row=19)
        self.label_wth.bind("<Button-1>",
                            lambda e: callback("https://en.wikipedia.org/wiki/Top-hat_transform"))
        self.wth = tk.Entry()
        self.wth.grid(column=1, row=21)
        self.label_seuil = tk.Label(
            text=f"Choisir le percentile minimum des points à conserver (99.2)\n Actuellement {seuil_pourcent}")
        self.label_seuil.grid(column=1, row=23)
        self.seuil = tk.Entry()
        self.seuil.grid(column=1, row=25)
        self.label_blob = tk.Label(text=f"Séléctionner les paramètres du blob detection")
        self.label_blob.grid(column=44, row=17)
        self.label_blob.bind("<Button-1>", lambda e: callback(
            "https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_log"))
        self.label_max = tk.Label(text=f"Entrer la valeur du max sigma (60)\n Actuellement {maximum_blob}")
        self.label_max.grid(column=44, row=19)
        self.max = tk.Entry()
        self.max.grid(column=44, row=21)
        self.label_num = tk.Label(text=f"Entrer la valeur du num sigma (10)\n Actuellement {num_blob}")
        self.label_num.grid(column=44, row=23)
        self.num = tk.Entry()
        self.num.grid(column=44, row=25)
        self.thresh = tk.Entry()
        self.label_thresh = tk.Label(text=f"Entrer la valeur du threshold (0.1)\n Actuellement {seuil_blob}")
        self.label_thresh.grid(column=44, row=27)
        self.thresh.grid(column=44, row=29)
        self.label_distance = tk.Label(
            text=f"Sélectionner la distance maximale entre les points semblables (3)\n Actuellement {distance_s}")
        self.label_distance.grid(column=1, row=27)
        self.distance = tk.Entry()
        self.distance.grid(column=1, row=29)
        self.bouton1 = tk.Button(text="Continuer analyse", command=self.continuer)
        self.bouton1.grid(column=20, row=42)

    def canal_oui(self, i):
        self.liste_canal.append(i)

    def continuer(self):
        for i in range(nombre_de_canaux):
            if self.liste[i].get() == 1:
                self.liste_canal.append(i)
        for i in range(nombre_de_zstack):
            if self.liste3[i].get() == 1:
                self.liste_z.append(i)
        mb.showinfo("Patience",
                    "L'analyse commencera quand vous cliquerez sur le (x) pour fermer la fenêtre «fenêtre czi». Lorsque c'est fait, patientez quelques instants")

        if self.sigma_gauss.get() == '':
            self.sigma_gauss = gauss
        else:
            self.sigma_gauss = int(self.sigma_gauss.get())

        if self.wth.get() == '':
            self.wth = th
        else:
            self.wth = int(self.wth.get())

        if self.seuil.get() == '':
            self.seuil = seuil_pourcent
        else:
            self.seuil = float(self.seuil.get())

        if self.max.get() == '':
            self.max = maximum_blob
        else:
            self.max = int(self.max.get())

        if self.num.get() == '':
            self.num = num_blob
        else:
            self.num = int(self.num.get())

        if self.thresh.get() == '':
            self.thresh = seuil_blob
        else:
            self.thresh = float(self.thresh.get())

        if self.distance.get() == '':
            self.distance = distance_s
        else:
            self.distance = int(self.distance.get())
        fichier_texte = open("text_file_czi.txt", "w")
        fichier_texte.write(str(self.sigma_gauss) + "\n")
        fichier_texte.write(str(self.wth) + "\n")
        fichier_texte.write(str(self.seuil) + "\n")
        fichier_texte.write(str(self.max) + "\n")
        fichier_texte.write(str(self.num) + "\n")
        fichier_texte.write(str(self.thresh) + "\n")
        fichier_texte.write(str(self.distance))
        fichier_texte.close()
        return self.distance, self.thresh, self.max, self.wth, self.seuil, self.sigma_gauss, self.num


beta=fenetre_analyse()
beta.title("Fenêtre czi")
