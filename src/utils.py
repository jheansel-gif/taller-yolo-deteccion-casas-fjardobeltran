import cv2
import matplotlib.pyplot as plt

def mostrar_imagen(ruta):

    img = cv2.imread(ruta)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.axis("off")
    plt.show()