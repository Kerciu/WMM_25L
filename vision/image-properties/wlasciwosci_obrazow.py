
import os
import sys
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt


skip_wnd = False


def printi(img, img_title="image"):
    """ Pomocnicza funkcja do wypisania informacji o obrazie. """
    print(f"{img_title}, wymiary: {img.shape}, typ danych: {img.dtype}, wartości: {img.min()} - {img.max()}")


def cv_imshow(img, img_title="image"):
    """
    Funkcja do wyświetlania obrazu w wykorzystaniem okna OpenCV.
    Wykonywane jest przeskalowanie obrazu z rzeczywistymi lub 16-bitowymi całkowitoliczbowymi wartościami pikseli,
    żeby jedną funkcją wywietlać obrazy różnych typów.
    """
    # cv2.namedWindow(img_title, cv2.WINDOW_AUTOSIZE) # cv2.WINDOW_NORMAL

    if (img.dtype == np.float32) or (img.dtype == np.float64):
        img_ = img / 255
    elif img.dtype == np.int16:
        img_ = img*128
    else:
        img_ = img
    cv2.imshow(img_title, img_)
    cv2.waitKey(1)  ### oczekiwanie przez bardzo krótki czas - okno się wyświetli, ale program się nie zablokuje, tylko będzie kontynuowany


""" Wczytanie obrazu z pliku """
image = cv2.imread("lena_mono.png", cv2.IMREAD_UNCHANGED)
printi(image, "image")

""" 
Obliczenie średniej bitowej dla pliku .png
os.stat() podaje rozmiar pliku w bajtach, a potrzebny jest w bitach (-> '8*')
"""
bitrate = 8*os.stat("lena_mono.png").st_size/(image.shape[0]*image.shape[1])
print(f"bitrate: {bitrate:.4f}")


""" 
Obliczanie entropii
"""

def calc_entropy(hist):
    pdf = hist/hist.sum() ### normalizacja histogramu -> rozkład prawdopodobieństwa; UWAGA: niebezpieczeństwo '/0' dla 'zerowego' histogramu!!!
    # entropy = -(pdf*np.log2(pdf)).sum() ### zapis na tablicach, ale problem z '/0'
    entropy = -sum([x*np.log2(x) for x in pdf if x != 0])
    return entropy


hist_image = cv2.calcHist([image], [0], None, [256], [0, 256])
""" 
cv2.calcHist() zwraca histogram w postaci tablicy 2D, 
do dalszego przetwarzania wygodniejsza może być tablica jednowymiarowa -> flatten().
"""
hist_image = hist_image.flatten()
# print(hist_image.sum(), 512*512) ### dla sprawdzenia: suma wartości histogramu powinna być równa liczbie pikseli w obrazie

H_image = calc_entropy(hist_image)
print(f"H(image) = {H_image:.4f}")


"""
Obraz różnicowy
"""


"""
Predykcja w kierunku poziomym:
od wartości danego piksela odejmowana jest wartość piksela z lewej strony - 'lewego sąsiada' (operacje na kolumnach).
Operację taką można wykonać dla pikseli leżących w drugiej i kolejnych kolumnach obrazu, z pominięciem skrajnie lewej kolumny.
"""
img_tmp1 = image[:, 1:]  ### wszystkie wiersze (':'), kolumny od 'pierwszej' do ostatniej ('1:')
img_tmp2 = image[:, :-1] ### wszystkie wiersze, kolumny od 'zerowej' do przedostatniej (':-1')

"""
W wyniku odejmowania pojawią się wartości ujemne - zakres wartości pikseli w obrazie różnicowym to będzie [-255, 255],
dlatego trzeba zminić typ wartości pikseli, żeby zakres wartości nie ograniczał się do [0, 255];
może to być np. cv2.CV_16S (odpowiednio np.int16 w NumPy), żeby pozostać w domenie liczb całkowitych.
"""
image_hdiff = cv2.addWeighted(img_tmp1, 1, img_tmp2, -1, 0, dtype=cv2.CV_16S)
printi(image_hdiff, "image_hdiff")
"""  
image_hdiff ma o jedną kolumnę mniej - dla skrajnie lewej kolumny nie było danych do odejmowania,
kolumnę tę można potraktować oddzielnie i 'połączyć' wyniki.
"""
image_hdiff_0 = cv2.addWeighted(image[:, 0], 1, 0, 0, -127, dtype=cv2.CV_16S) ### od 'zerowej' kolumny obrazu oryginalnego odejmowana stała wartość '127'
printi(image_hdiff_0, "image_hdiff_0")
image_hdiff = np.hstack((image_hdiff_0, image_hdiff)) ### połączenie tablic w kierunku poziomym, czyli 'kolumna za kolumną'
printi(image_hdiff, "image_hdiff")

"""
Funkcja cv2.imshow() zakłada inny zakres wartości pikseli w zależności od typu danych;
żeby uzyskać poprawne wyświetlanie obrazów z 16-bitowymi wartościami pikseli należy wartości pikseli
pomnożyć przez 128 (zmiana zakresu z [-255, 255] na [-32640, 32640], co pokrywa prawie cały zakres wartości dostępnych 
dla 16-bitowych liczb całkowitych).
"""
cv2.imshow("image_hdiff not scaled", image_hdiff) ### obraz 'całkowicie szary', co odpowiada wartościom bliskim 0
cv2.imshow("image_hdiff scaled", image_hdiff*128) ### obraz z widocznymi zmianami, czerń - wartość minimalna (ujemna), szrość - poziom 0, biel - wartość maksymalna
cv_imshow(image_hdiff, "image_hdiff")             ### zdefiniowana funkcja pomocnicza odpowiednio 'obsługuje' obrazy z 16-bitowymi wartościami

"""  
Predykcja w kierunku pionowym: 
od wartości danego piksela odejmowana jest wartość 'górnego sąsiada';
realizacja podobnie jak powyżej, ale operacje na wierszach.
"""
image_vdiff = cv2.addWeighted(image[1:, :], 1, image[:-1, :], -1, 0, dtype=cv2.CV_16S)
printi(image_vdiff, "image_vdiff")
"""
image_vdiff ma o jedną linię mniej - dla skrajnie górnej linii nie było danych do odejmowania,
wiersz ten można potraktować oddzielnie i 'połączyć' wyniki.
UWAGA: pobranie 'zerowego' wiersza w postaci [0:1, :] (a nie po prostu: [0, :]) żeby wymusić tablicę 2D 
(zapis [0, :] zwróci tablicę 1D, która będzie traktowana jak kolumna i pojawią się komunikaty o 'niezgodności wymiarów').
"""
image_vdiff_0 = cv2.addWeighted(image[0:1, :], 1, 0, 0, -127, dtype=cv2.CV_16S)
printi(image_vdiff_0, "image_vdiff_0")
image_vdiff = np.vstack((image_vdiff_0, image_vdiff)) ### połączenie tablic w kierunku pionowym, czyli 'wiersz za wierszem'
printi(image_vdiff, "image_vdiff")
cv_imshow(image_vdiff, "image_vdiff")

if not skip_wnd: cv2.waitKey(0)          ### oczekiwanie na reakcję użytkownika - naciśnięcie klawisza lub zamknięcie okien
cv2.destroyAllWindows() ### zniszczenie okien (niezamknięcie okien przez zakończeniem skryptu może skutkować błędem przy próbie ich zamknięcia później)

""" Entropia dla obrazów różnicowych """

"""
cv2.calcHist() wymaga danych w formacie liczb całkowitych bez znaku (8- lub 16-bitowych) lub 32-bitowych liczb rzeczywistych, 
dlatego wartości pikseli są przesuwane z zakresu [-255, 255] do [0, 510] (-> '+255') 
oraz konwertowane na typ np.uint16 (-> astype()).
"""
image_tmp = (image_hdiff+255).astype(np.uint16)
hist_hdiff = cv2.calcHist([image_tmp], [0], None, [511], [0, 511]).flatten()
# print(hist_hdiff.sum())

H_hdiff = calc_entropy(hist_hdiff)
print(f"H(hdiff) = {H_hdiff:.4f}")

image_tmp = (image_vdiff+255).astype(np.uint16)
hist_vdiff = cv2.calcHist([image_tmp], [0], None, [511], [0, 511]).flatten()

H_vdiff = calc_entropy(hist_vdiff)
print(f"H(vdiff) = {H_vdiff:.4f}")

""" Wyświetlenie histogramów z wykorzystaniem matplotlib.pyplot """
plt.figure()
plt.plot(hist_image, color="blue")
plt.title("hist_image")
plt.xlim([0, 255])
plt.figure()
plt.plot(np.arange(-255, 256, 1), hist_hdiff, color="red") ### jawne podane wartości 'x' i 'y', żeby zmienić opisy na osi poziomej
plt.title("hist_hdiff")
plt.xlim([-255, 255])
plt.figure()
plt.plot(np.arange(-255, 256, 1), hist_vdiff, color="red")
plt.title("hist_vdiff")
plt.xlim([-255, 255])
if not skip_wnd: plt.show() ### wyświetlenie okien i oczekiwanie na ich zamnkięcie


""" 
Transformacja falkowa obrazu
"""


def dwt(img):
    """
    Bardzo prosta i podstawowa implementacja, nie uwzględniająca efektywnych metod obliczania DWT
    i dopuszczająca pewne niedokładności.
    """
    maskL = np.array([0.02674875741080976, -0.01686411844287795, -0.07822326652898785, 0.2668641184428723,
        0.6029490182363579, 0.2668641184428723, -0.07822326652898785, -0.01686411844287795, 0.02674875741080976])
    maskH = np.array([0.09127176311424948, -0.05754352622849957, -0.5912717631142470, 1.115087052456994,
        -0.5912717631142470, -0.05754352622849957, 0.09127176311424948])

    bandLL = cv2.sepFilter2D(img,         -1, maskL, maskL)[::2, ::2]
    bandLH = cv2.sepFilter2D(img, cv2.CV_16S, maskL, maskH)[::2, ::2] ### ze względu na filtrację górnoprzepustową -> wartości ujemne, dlatego wynik 16-bitowy ze znakiem
    bandHL = cv2.sepFilter2D(img, cv2.CV_16S, maskH, maskL)[::2, ::2]
    bandHH = cv2.sepFilter2D(img, cv2.CV_16S, maskH, maskH)[::2, ::2]

    return bandLL, bandLH, bandHL, bandHH


ll, lh, hl, hh = dwt(image)
printi(ll, "LL")
printi(lh, "LH")
printi(hl, "HL")
printi(hh, "HH")

cv_imshow(ll, "LL2")
cv_imshow(cv2.multiply(lh, 2), "LH2") ### cv2.multiply() -> zwiększenie kontrastu obrazów 'H', żeby lepiej uwidocznić
cv_imshow(cv2.multiply(hl, 2), "HL2")
cv_imshow(cv2.multiply(hh, 2), "HH2")

""" Entropia dla obrazów pasmowych """

hist_ll = cv2.calcHist([ll], [0], None, [256], [0, 256]).flatten()
hist_lh = cv2.calcHist([(lh+255).astype(np.uint16)], [0], None, [511], [0, 511]).flatten() ### zmiana zakresu wartości i typu danych ze względu na cv2.calcHist() (jak wcześniej przy obrazach różnicowych)
hist_hl = cv2.calcHist([(hl+255).astype(np.uint16)], [0], None, [511], [0, 511]).flatten()
hist_hh = cv2.calcHist([(hh+255).astype(np.uint16)], [0], None, [511], [0, 511]).flatten()
H_ll = calc_entropy(hist_ll)
H_lh = calc_entropy(hist_lh)
H_hl = calc_entropy(hist_hl)
H_hh = calc_entropy(hist_hh)
print(f"H(LL) = {H_ll:.4f} \nH(LH) = {H_lh:.4f} \nH(HL) = {H_hl:.4f} \nH(HH) = {H_hh:.4f} \nH_śr = {(H_ll+H_lh+H_hl+H_hh)/4:.4f}")

""" Wyświetlenie histogramów - jeden obraz z czterema pod-obrazami """
fig = plt.figure()
fig.set_figheight(fig.get_figheight()*2) ### zwiększenie rozmiarów okna
fig.set_figwidth(fig.get_figwidth()*2)
plt.subplot(2, 2, 1)
plt.plot(hist_ll, color="blue")
plt.title("hist_ll")
plt.xlim([0, 255])
plt.subplot(2, 2, 3)
plt.plot(np.arange(-255, 256, 1), hist_lh, color="red")
plt.title("hist_lh")
plt.xlim([-255, 255])
plt.subplot(2, 2, 2)
plt.plot(np.arange(-255, 256, 1), hist_hl, color="red")
plt.title("hist_hl")
plt.xlim([-255, 255])
plt.subplot(2, 2, 4)
plt.plot(np.arange(-255, 256, 1), hist_hh, color="red")
plt.title("hist_hh")
plt.xlim([-255, 255])

if not skip_wnd: plt.show() ### wyświetlenie okien i oczekiwanie na ich zamnkięcie
cv2.destroyAllWindows() ### zniszczenie okien


""" Obraz barwny """


""" Entropia dla składowych obrazu barwnego. """

image_col = cv2.imread("lena_col.png")
printi(image_col, "image_col")

image_R = image_col[:, :, 2] ### cv2.imread() zwraca obrazy w formacie BGR
image_G = image_col[:, :, 1]
image_B = image_col[:, :, 0]

hist_R = cv2.calcHist([image_R], [0], None, [256], [0, 256]).flatten()
hist_G = cv2.calcHist([image_G], [0], None, [256], [0, 256]).flatten()
hist_B = cv2.calcHist([image_B], [0], None, [256], [0, 256]).flatten()

H_R = calc_entropy(hist_R)
H_G = calc_entropy(hist_G)
H_B = calc_entropy(hist_B)
print(f"H(R) = {H_R:.4f} \nH(G) = {H_G:.4f} \nH(B) = {H_B:.4f} \nH_śr = {(H_R+H_G+H_B)/3:.4f}")

cv_imshow(image_R, "image_R")
cv_imshow(image_G, "image_G")
cv_imshow(image_B, "image_B")
plt.figure()
plt.plot(hist_R, color="red")
plt.plot(hist_G, color="green")
plt.plot(hist_B, color="blue")
plt.title("hist RGB")
plt.xlim([0, 255])
if not skip_wnd: plt.show()              ### wyświetlenie okien i oczekiwanie na ich zamnkięcie
cv2.destroyAllWindows() ### zniszczenie również okien OpenCV

""" Konwersja do YCrCb. """

image_YCrCb = cv2.cvtColor(image_col, cv2.COLOR_BGR2YCrCb) ### albo: cv2.COLOR_BGR2YUV
printi(image_YCrCb, "image_YCrCb")

hist_Y = cv2.calcHist([image_YCrCb[:, :, 0]], [0], None, [256], [0, 256]).flatten()
hist_Cr = cv2.calcHist([image_YCrCb[:, :, 1]], [0], None, [256], [0, 256]).flatten()
hist_Cb = cv2.calcHist([image_YCrCb[:, :, 2]], [0], None, [256], [0, 256]).flatten()

H_Y = calc_entropy(hist_Y)
H_Cr = calc_entropy(hist_Cr)
H_Cb = calc_entropy(hist_Cb)
print(f"H(Y) = {H_Y:.4f} \nH(Cr) = {H_Cr:.4f} \nH(Cb) = {H_Cb:.4f} \nH_śr = {(H_Y+H_Cr+H_Cb)/3:.4f}")

cv_imshow(image_YCrCb[:, :, 0], "image_Y")
cv_imshow(image_YCrCb[:, :, 1], "image_Cr")
cv_imshow(image_YCrCb[:, :, 2], "image_Cb")
plt.figure()
plt.plot(hist_Y, color="gray")
plt.plot(hist_Cr, color="red")
plt.plot(hist_Cb, color="blue")
plt.title("hist YCrCb")
plt.xlim([0, 255])
if not skip_wnd: plt.show()              ### wyświetlenie okien i oczekiwanie na ich zamnkięcie
cv2.destroyAllWindows() ### zniszczenie również okien OpenCV


""" Wyznaczanie charakterystyki R-D """


def calc_mse_psnr(img1, img2):
    """ Funkcja obliczająca MSE i PSNR dla różnicy podanych obrazów, zakładana wartość pikseli z przedziału [0, 255]. """

    imax = 255.**2 ### maksymalna wartość sygnału -> 255
    """
    W różnicy obrazów istotne są wartości ujemne, dlatego img1 konwertowany jest do typu np.float64 (liczby rzeczywiste) 
    aby nie ograniczać wyniku do przedziału [0, 255].
    """
    mse = ((img1.astype(np.float64)-img2)**2).sum()/img1.size ###img1.size - liczba elementów w img1, ==img1.shape[0]*img1.shape[1] dla obrazów mono, ==img1.shape[0]*img1.shape[1]*img1.shape[2] dla obrazów barwnych
    psnr = 10.0*np.log10(imax/mse)
    return (mse, psnr)


image = cv2.imread("lena_col.png", cv2.IMREAD_UNCHANGED)
xx = [] ### tablica na wartości osi X -> bitrate
ym = [] ### tablica na wartości osi Y dla MSE
yp = [] ### tablica na wartości osi Y dla PSNR

for quality in [90, 50, 10]: ### wartości dla parametru 'quality' należałoby dobrać tak, aby uzyskać 'gładkie' wykresy...
    out_file_name = f"out_image_q{quality:03d}.jpg"
    """ Zapis do pliku w formacie .jpg z ustaloną 'jakością' """
    cv2.imwrite(out_file_name, image, (cv2.IMWRITE_JPEG_QUALITY, quality))
    """ Odczyt skompresowanego obrazu, policzenie bitrate'u i PSNR """
    image_compressed = cv2.imread(out_file_name, cv2.IMREAD_UNCHANGED)
    bitrate = 8*os.stat(out_file_name).st_size/(image.shape[0]*image.shape[1]) ### image.shape == image_compressed.shape
    mse, psnr = calc_mse_psnr(image, image_compressed)
    """ Zapamiętanie wyników do pózniejszego wykorzystania """
    xx.append(bitrate)
    ym.append(mse)
    yp.append(psnr)

""" Narysowanie wykresów """
fig = plt.figure()
fig.set_figwidth(fig.get_figwidth()*2)
plt.suptitle("Charakterystyki R-D")
plt.subplot(1, 2, 1)
plt.plot(xx, ym, "-.")
plt.title("MSE(R)")
plt.xlabel("bitrate")
plt.ylabel("MSE", labelpad=0)
plt.subplot(1, 2, 2)
plt.plot(xx, yp, "-o")
plt.title("PSNR(R)")
plt.xlabel("bitrate")
plt.ylabel("PSNR [dB]", labelpad=0)
plt.show()


cv2.waitKey(0)          ### oczekiwanie na reakcję użytkownika
cv2.destroyAllWindows() ### należy pamiętać o zniszczeniu okien na końcu programu
