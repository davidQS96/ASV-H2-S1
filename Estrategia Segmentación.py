import numpy as np
import cv2
import matplotlib.pyplot as plt


#Inicialización de imagen

imagecol = cv2.imread('bolt1.png')
result = imagecol.copy()
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
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


#Parte pista

#Segmatación por matriz
lower_blue = np.array([0, 70, 40])
upper_blue = np.array([8,130,200])
mask2 = cv2.inRange(hsv,lower_blue,upper_blue)

#Deteción de contronos
contours, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
mask22 = cv2.inRange(hsv2,lower_blue,upper_blue)
maskf2 = mask22 + mask2
maskf2c = maskf2.copy()

#Eliminación de objetos no deseados
maskf2[0:400,:] = 0 

marker = np.zeros_like(imagecol)
contours, _ = cv2.findContours(maskf2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#Detección de objeto y ubicación de pistas
for cnt in contours:
    area = cv2.contourArea(cnt)
    approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
    x = approx.ravel()[0]
    y = approx.ravel()[1]


    if area > 200:
        cv2.drawContours(marker, [approx], 0, (255, 255, 255), cv2.FILLED)
marker = cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)
maskf2 = maskf2-~marker







#Combinación de mascaras
maskf1 = 2 <= maskf1
maskf2 = 2 <= maskf2
mask = np.logical_or(maskf1,maskf2)
cmask = mask.copy()
mask = ~mask
maskf2c = 2 <= maskf2c
#maskf2c[401:h,:]=0


# Reducción de imagen por analisis
cuarto = int(h/3)
mask[0:cuarto][:] = 0
mask[h-1-cuarto:h-1][:] = 0






#marker[:,:,0] = marker[:,:,0]+mask

#Aplicación de mascara
final = imagecol.copy()
final[:,:,0] = final[:,:,0]*mask
final[:,:,1] = final[:,:,1]*mask
final[:,:,2] = final[:,:,2]*mask

mask = np.logical_xor(mask,maskf2c)
final[:,:,0] = final[:,:,0]*mask
final[:,:,1] = final[:,:,1]*mask
final[:,:,2] = final[:,:,2]*mask

lower_blue = np.array([60, 70, 0])
upper_blue = np.array([359,255,255])
maskd = cv2.inRange(hsv,lower_blue,upper_blue)



# Umbralización
lim = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
limc = lim.copy()


lim = 180 <= lim
mask = np.logical_xor(mask,lim)


final[:,:,0] = final[:,:,0]*mask
final[:,:,1] = final[:,:,1]*mask
final[:,:,2] = final[:,:,2]*mask

#Reconstrucción de la imagen
marker = np.zeros_like(imagecol)
contours, _ = cv2.findContours(limc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#Detección de objeto y ubicación de pistas
for cnt in contours:
    area = cv2.contourArea(cnt)
    approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)

    if area > 200:
        cv2.drawContours(marker, contours, -1, (255, 255, 255), cv2.FILLED)
marker = cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(final, cv2.COLOR_BGR2HSV)
lower_blue = np.array([0, 70, 50])
upper_blue = np.array([360,130,120])
maskd = cv2.inRange(hsv,lower_blue,upper_blue)

marker = 180 <= marker

result[:,:,0] = gray*~marker+result[:,:,0]*marker
result[:,:,1] = gray*~marker+result[:,:,1]*marker
result[:,:,2] = gray*~marker+result[:,:,2]*marker


plt.imshow(marker,cmap='gray',vmin=0,vmax=1)
cv2.imwrite('imagen.png',final)
cv2.imwrite('imagen2.png',result)
