import matplotlib.pyplot as plt
import cv2

def mostrar_imagen(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis("off")