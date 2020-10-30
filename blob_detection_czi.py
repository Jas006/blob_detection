#from fenetre_czi import *
import numpy as np
import cv2
from scipy import ndimage
from matplotlib import pyplot as plt
from skimage.feature import blob_log
from skimage.color import rgb2gray
import czifile
from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import white_tophat as hat
from skimage.morphology import reconstruction
from scipy.optimize import leastsq
import xlsxwriter
from PIL import Image


def complet_czi(path,channel,liste_z,sigma_filtre_gaussien=3,taille_tophat=30,seuil_percentile=99.2,max_sigma=30, num_sigma=10, threshold=0.1,Path_enregistrement=None,d_semblables=3,enregistrementblob=None,enregistrement_blur=None,enregistrement_hdome=None,enregistrement_th=None,enregistrement_seuil=None,oui_non_image1="non",enregistrement_image1=None,enregistrement_detection=None):
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
        im=Image.fromarray(image_debruite)
        im.save(enregistrement_blur+fr"\channel_{gamma+1}_z_{zeta+1}.tif")
        return image_debruite
    def h_dome(image_debruite,gamma,zeta):
        seed = np.copy(image_debruite)
        seed[1:-1, 1:-1] = image_debruite.min()
        mask = image_debruite
        dilated = reconstruction(seed, mask, method='dilation')
        image_a_analyser = image_debruite - dilated
        plt.imshow(image_a_analyser)
        plt.savefig(enregistrement_hdome + rf"\channel_{gamma+1}_z_{zeta+1}.png")
        return image_a_analyser
    def top_hat(image_a_analyser,gamma,zeta):
        image_a_analyser = hat(image_a_analyser, taille_tophat)
        plt.imshow(image_a_analyser)
        plt.savefig(enregistrement_th + rf"\channel_{gamma+1}_z_{zeta+1}.png")
        return image_a_analyser
    def thresholding(image_a_analyser,gamma,zeta):
        minimum = np.percentile(image_a_analyser, seuil_percentile)
        for i in range(x_image):
            for j in range(y_image):
                if image_a_analyser[i][j] <= minimum:
                    image_a_analyser[i][j] = 0
        plt.imshow(image_a_analyser)
        plt.savefig(enregistrement_seuil + f"\channel_{gamma+1}_z_{zeta+1}.png")
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
        worksheet = liste_excel[somme_excel]
        worksheet.write('A1', 'x', bold)
        worksheet.write('B1', 'y', bold)
        liste_matrices=[]
        def matrix(ly, lx):
            matriceb = np.zeros((2048, 2048))
            for i in range(-1*len(channel), 1+len(channel)):
                for k in range(-1*len(channel), 1+len(channel)):
                    for j in range(len(ly)):
                        if i + ly[j] >= 100 or i + ly[j] < 0 or k + lx[j] >= 100 or k + lx[j] < 0:
                            pass
                        else:
                            matriceb[ly[j] + i, lx[j] + k] = 1
            return matriceb
        for m in range(len(channel)):
            liste_matrices.append(matrix(liste_gauss_y[m],liste_gauss_x[m]))
        matrice_grosse=np.zeros((y_image,x_image))
        for n in range(len(channel)):
            matrice_grosse+=liste_matrices[n]
        x = np.where(matrice_grosse > len(channel)-1)[1]
        y = np.where(matrice_grosse > len(channel)-1)[0]
        derni = np.zeros((y_image, x_image))
        derni[y, x] = 1
        frame = cv2.inRange(derni, 1, 10)
        blobs = frame > 10
        labels, nlabels = ndimage.label(blobs)
        # find the center of mass of each label
        t = ndimage.center_of_mass(frame, labels, np.arange(nlabels) + 1)

        for i in range(len(t)):
            worksheet.write(1+i,1,t[1])
            worksheet.write(1+i,2,t[0])
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
        offset = abs(np.median(data))
        y = (Y * data).sum() / total
        col = data[:, int(y)]
        sigma = np.sqrt(np.abs((np.arange(col.size) - x) ** 2 * col).sum() / col.sum())
        height = abs(data.max())
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
            if oui_non_image1 == 'oui':
                image1=Image.fromarray(image)
                image1.save(enregistrement_image1+fr"\z{z+1}_channel_{c+1}.tif")
            else:
                pass
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
            matrice=np.zeros((x_image,y_image))
            for n in range(len(liste_gauss_x)):
                liste_gauss_x[sommec][n]=int(liste_gauss_x[sommec][n])
                liste_gauss_y[sommec][n] = int(liste_gauss_y[sommec][n])
                matrice[liste_gauss_y[sommec][n],liste_gauss_x[sommec][n]]=1
            image_finale=Image.fromarray(matrice)
            image_finale.save(enregistrement_detection + fr"\canal{channel[c]+1}_z{liste_z[z]+1}.tif")
            np.save(enregistrement_detection + fr"\canal{channel[c]+1}_z{liste_z[z]+1}",matrice)
            worksheet=liste_excel[sommej]
            worksheet.write('F1', 'x', bold)
            worksheet.write('G1', 'y', bold)
            worksheet.write('H1', 'sigma', bold)
            worksheet.write('I1', 'offset', bold)
            worksheet.write('J1', 'intensité_z', bold)
            worksheet.write('K1', 'integrale', bold)
            sommeg=0
            for i in range(len(liste_gauss_x[sommec])):
                sommeg+=1
                data, min_x, min_y = reduire_image(image_importe[0, 0, channel[c], 0, liste_z[z], :, :, 0], liste_gauss_x[sommec][i],
                                                   liste_gauss_y[sommec][i], 15, 15)
                params = fitgaussian(data)
                fit = gaussian(*params)
                (height, x, y, sigma, offset) = params

                integrale = 2 * np.pi * height * sigma ** 2
                if integrale<1000000000:
                    worksheet.write(sommeg, 5, x + min_x)
                    worksheet.write(sommeg, 6, y + min_y)
                    worksheet.write(sommeg, 7, sigma)
                    worksheet.write(sommeg, 8, abs(offset))
                    worksheet.write(sommeg, 9, abs(height))
                    worksheet.write(sommeg, 10, integrale)
                else:
                    success = workbook.add_format(
                        {
                            "bg_color": "#FF0000"
                        }
                    )

                    worksheet.write(sommeg, 5, x + min_x,success)
                    worksheet.write(sommeg, 6, y + min_y,success)
                    worksheet.write(sommeg, 7, sigma,success)
                    worksheet.write(sommeg, 8, abs(offset),success)
                    worksheet.write(sommeg, 9, abs(height),success)
                    worksheet.write(sommeg, 10, integrale,success)
            sommec+=1
            sommej += 1
        sommej+=1


    workbook.close()