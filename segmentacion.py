import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import io

#Inicialización de imagen

imagecol = cv2.imread('bolt1.jpg') 
marker = np.zeros_like(imagecol)
hsv = cv2.cvtColor(imagecol, cv2.COLOR_BGR2HSV)

#Parte cesped

#Segmatación por matriz
lower_blue = np.array([30, 70, 50])
upper_blue = np.array([60,130,200])
mask1 = cv2.inRange(hsv,lower_blue,upper_blue)
contours, _ = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


#Deteción de contronos
cv2.drawContours(marker,contours, -1, (255, 255, 255), 5)
h, w = marker.shape[:2]
maskll = np.zeros((h+2, w+2), np.uint8)
 

#Relleno de contornos      
img_pl = np.zeros_like(imagecol)
cv2.fillPoly(img_pl,pts=contours,color=(255,255,255))
#img_pl = ~img_pl
alt = img_pl.copy()
cv2.floodFill(img_pl, maskll, (0,0), 255);


#Relleno de huecos
hsv2 = cv2.cvtColor(img_pl, cv2.COLOR_BGR2HSV)
lower_blue = np.array([0, 0, 0])
upper_blue = np.array([360,20,100])
mask11 = cv2.inRange(hsv2,lower_blue,upper_blue)
maskf1 = mask11 + mask1


#Parte pista repite los pasos anteriores

lower_blue = np.array([0, 70, 40])
upper_blue = np.array([8,130,200])
mask2 = cv2.inRange(hsv,lower_blue,upper_blue)

contours, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(marker,contours, -1, (255, 255, 255), 5)
h, w = marker.shape[:2]
maskll = np.zeros((h+2, w+2), np.uint8)
       
img_pl = np.zeros_like(imagecol)
cv2.fillPoly(img_pl,pts=contours,color=(255,255,255))
#img_pl = ~img_pl
alt = img_pl.copy()
cv2.floodFill(img_pl, maskll, (0,0), 255);

hsv2 = cv2.cvtColor(img_pl, cv2.COLOR_BGR2HSV)
lower_blue = np.array([0, 0, 0])
upper_blue = np.array([360,20,100])
mask22 = cv2.inRange(hsv2,lower_blue,upper_blue)
maskf2 = mask22 + mask2



#Combinación de mascaras
mask = maskf1+maskf2
mask = ~mask

# Reducción de imagen por analisis
cuarto = int(h/3)
mask[0:cuarto][:] = 0
mask[h-1-cuarto:h-1][:] = 0


#Aplicación de mascara
final = imagecol.copy()
final[:,:,0] = final[:,:,0]*mask
final[:,:,1] = final[:,:,1]*mask
final[:,:,2] = final[:,:,2]*mask
contours, _ = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
plt.imshow(final,cmap='gray',vmin=0,vmax=255)
io.imsave('imagen.png',final)