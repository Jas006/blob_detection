#from fenetre_czi import *
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



def complet_czi(path,channel,liste_z,sigma_filtre_gaussien=3,taille_tophat=30,seuil_percentile=99.2,max_sigma=30, num_sigma=10, threshold=0.1,Path_enregistrement=None,d_semblables=3,enregistrementblob=None,enregistrement_blur=None,enregistrement_hdome=None,enregistrement_th=None,enregistrement_seuil=None):
    workbook = xlsxwriter.Workbook(Path_enregistrement+r'\analyze_excel.xlsx')
    liste_gauss_x=[]
    liste_gauss_y=[]
    liste_excel = []
    bold = workbook.add_format({'bold': True})
    image_importe = czifile.imread(path)
    x_image=len(image_importe[0,0,0,0,0])
    y_image=len(image_importe[0,0,0,0,0,0])
    for z in range(len(liste_z)):
        for c in range(len(channel)):
            liste_excel.append(workbook.add_worksheet(f"Canal{c+1}_z{liste_z[z]+1}"))
        liste_excel.append(workbook.add_worksheet(f"Points semblables z{liste_z[z]+1}"))
    for i in range(len(liste_excel)):
        liste_excel[i].set_column('A:A',20)

    def denoising(image,gamma,zeta):
        # On applique un filtre (gaussien, circulaire, moyen...) sur l'image pour enlever le bruit
        image_debruite = gaussian_filter(image, sigma_filtre_gaussien)
        plt.imshow(image_debruite,cmap="gray")
        plt.savefig(enregistrement_blur+fr"\channel_{gamma}_z_{zeta}.png")
        return image_debruite
    def h_dome(image_debruite,gamma,zeta):
        seed = np.copy(image_debruite)
        seed[1:-1, 1:-1] = image_debruite.min()
        mask = image_debruite
        dilated = reconstruction(seed, mask, method='dilation')
        image_a_analyser = image_debruite - dilated
        plt.imshow(image_a_analyser)
        plt.savefig(enregistrement_hdome + rf"\channel_{gamma}_z_{zeta}.png")
        return image_a_analyser
    def top_hat(image_a_analyser,gamma,zeta):
        image_a_analyser = hat(image_a_analyser, taille_tophat)
        plt.imshow(image_a_analyser)
        plt.savefig(enregistrement_th + rf"\channel_{gamma}_z_{zeta}.png")
        return image_a_analyser
    def thresholding(image_a_analyser,gamma,zeta):
        minimum = np.percentile(image_a_analyser, seuil_percentile)
        for i in range(x_image):
            for j in range(y_image):
                if image_a_analyser[i][j] <= minimum:
                    image_a_analyser[i][j] = 0
        plt.imshow(image_a_analyser)
        plt.savefig(enregistrement_seuil + f"\channel_{gamma}_z_{zeta}.png")
        return image_a_analyser
    def blob_detection(image,gamma,zeta,somme_excel,liste):
        image=rgb2gray(image)

        blobs_log=blob_log(image,max_sigma=max_sigma,num_sigma=num_sigma,threshold=threshold)

        fig, ax = plt.subplots(1, 1, figsize=(9, 9), sharex=True, sharey=True)
        ax.imshow(image, interpolation='nearest')
        somme = 0

        worksheet=liste_excel[somme_excel]
        worksheet.write('A1', 'x', bold)
        worksheet.write('B1', 'y', bold)
        worksheet.write('C1', 'r', bold)
        worksheet.write('D1', 'I', bold)
        liste11 = []
        liste12 = []
        sommes = 1

        for blob in blobs_log:

            y, x, r = blob
            luminosite_point = image[int(y)][int(x)]
            if luminosite_point > 0:
                if r >= 1:
                    liste11.append(x)
                    liste12.append(y)

            if r >= 1 and luminosite_point > 0:
                somme += 1
                c = plt.Circle((x, y), r, color='yellow', linewidth=2, fill=False)
                ax.add_patch(c)

                worksheet.write(sommes, 0, x)
                worksheet.write(sommes, 1, y)
                worksheet.write(sommes, 2, r)
                worksheet.write(sommes, 3, luminosite_point)
                sommes += 1

        plt.savefig(enregistrementblob+rf"\z{zeta+1}_channel{gamma+1}")
        somme_excel+=1
        liste.append(liste11)
        liste.append(liste12)
        liste_gauss_x.append(liste11)
        liste_gauss_y.append(liste12)
    def points_semblables(liste, somme_excel):
        la_matrice_des_points_semblables = np.zeros((x_image, y_image))
        worksheet = liste_excel[somme_excel]
        worksheet.write('A1', 'x', bold)
        worksheet.write('B1', 'y', bold)
        carre_d = d_semblables ** 2
        for i in range(3):
            for j in range(3):
                for k in range(len(liste[i*2])):
                    for l in range(len(liste[j*2])):
                        x1 = int(liste[i * 2][k])
                        x2 = int(liste[j * 2][l])
                        y1 = int(liste[i * 2 + 1][k])
                        y2 = int(liste[j * 2 + 1][l])
                        if (x1 - x2)**2 +  (y1-y2)**2 < carre_d:
                            la_matrice_des_points_semblables[x1][y1] += 1

        Somme = 0
        for i in range(x_image):
            for j in range(y_image):
                    if la_matrice_des_points_semblables[i][j]==len(channel)*2:
                        worksheet.write(Somme + 1, 0, i)
                        worksheet.write(Somme + 1, 1, j)
                        Somme += 1
        return Somme
    def gaussian(height, center_x, center_y, sigma, offset):
        """Returns a gaussian function with the given parameters"""
        sigma = float(sigma)
        return lambda x, y: height * np.exp(
            -(((center_x - x) / sigma) ** 2 + ((center_y - y) / sigma) ** 2) / 2) + offset
    def moments(data):
        """Returns (height, x, y,sigma)
        the gaussian parameters of a 2D distribution by calculating its
        moments """
        total = data.sum()
        X, Y = np.indices(data.shape)
        x = (X * data).sum() / total
        offset = np.median(data)
        y = (Y * data).sum() / total
        col = data[:, int(y)]
        sigma = np.sqrt(np.abs((np.arange(col.size) - x) ** 2 * col).sum() / col.sum())
        height = data.max()
        return height, x, y, sigma, offset
    def fitgaussian(data):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution found by a fit"""
        params = moments(data)
        errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                           data)
        p, success = leastsq(errorfunction, params)
        return p
    def reduire_image(image, point_x, point_y, largeur_x, largeur_y):
        while point_x - largeur_x < 0:
            if point_x - largeur_x < 0:
                largeur_x -= 1
        while point_x + largeur_x >= x_image:
            if point_x + largeur_x >= x_image:
                largeur_x -= 1
        while point_y - largeur_y < 0:
            if point_y - largeur_y < 0:
                largeur_y -= 1
        while point_y + largeur_y >= y_image:
            if point_y + largeur_y >= y_image:
                largeur_y -= 1
        min_x = point_x - largeur_x
        min_y = point_y - largeur_y

        data = image[int(point_y - largeur_y):int(point_y + largeur_y),
               int(point_x - largeur_x):int(point_x + largeur_x)]
        return data, min_x, min_y

    sommep=0
    for z in range(len(liste_z)):
        liste_points = []
        liste_filtre=[]

        for c in range(len(channel)):
            image=image_importe[0,0,channel[c],0,liste_z[z],:,:,0]
            image_denoising=denoising(image,channel[c],liste_z[z])
            image_hdome=h_dome(image_denoising,channel[c],liste_z[z])
            image_top_hat=top_hat(image_hdome,channel[c],liste_z[z])
            image_threshold=thresholding(image_top_hat,channel[c],liste_z[z])
            liste_filtre.append(image_threshold)
        if len(channel) > 1:
            for c in range(len(channel)+1):
                if (sommep + 1) % (len(channel) + 1) != 0:
                    blob_detection(liste_filtre[c], gamma=channel[c], zeta=liste_z[z], somme_excel=sommep,
                                   liste=liste_points)
                    sommep += 1
                else:
                    points_semblables(liste_points, sommep)
                    sommep += 1
        else:
            for c in range(len(channel)):
                blob_detection(liste_filtre[c], gamma=channel[c], zeta=liste_z[z], somme_excel=sommep,
                               liste=liste_points)
                sommep += 1

    sommej=0
    sommec=0
    for z in range(len(liste_z)):
        for c in range(len(channel)):
            worksheet=liste_excel[sommej]
            worksheet.write('F1', 'x', bold)
            worksheet.write('G1', 'y', bold)
            worksheet.write('H1', 'sigma', bold)
            worksheet.write('I1', 'offset', bold)
            worksheet.write('J1', 'integrale', bold)
            sommeg=0
            for i in range(len(liste_gauss_x[sommec])):
                sommeg+=1
                data, min_x, min_y = reduire_image(image_importe[0, 0, channel[c], 0, liste_z[z], :, :, 0], liste_gauss_x[sommec][i],
                                                   liste_gauss_y[sommec][i], 15, 15)
                params = fitgaussian(data)
                fit = gaussian(*params)
                (height, x, y, sigma, offset) = params

                integrale = 2 * np.pi * height * sigma ** 2
                worksheet.write(sommeg, 5, x + min_x)
                worksheet.write(sommeg, 6, y + min_y)
                worksheet.write(sommeg, 7, sigma)
                worksheet.write(sommeg, 8, offset)
                worksheet.write(sommeg, 9, integrale)
            sommec+=1
            sommej += 1
        sommej+=1


    workbook.close()