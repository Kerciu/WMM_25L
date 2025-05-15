# -*- coding: utf-8 -*-
"""
## *Nagłówek sprawozdania*

w sprawozdaniu należy zamieścić *kto, kiedy, na jakich zajęciach, itp., itd.*!
"""

"""
# Analiza obrazu - detekcja twarzy

Do realizacji zadań wykorzystywane są biblioteki OpenCV, dlib oraz InsightFace (wymaga dodatkowo pakietu onnxruntime). 
Biblioteki te muszą być zainstalowane w środowisku Python, odpowiedni plik 'requirements.txt' został udostępniony razem ze skryptem.

Do wykonania skryptu potrzebne są:
- obraz testowy: '___.jpg'
- model detektora Haar: 'haarcascade_frontalface_default.xml'
- model detektora MMOD (sieć neuronowa z biblioteki dlib): 'mmod_human_face_detector.dat'
- modele detektora twarzy (siec neuronowa) z biblioteki insightface: katalog 'insightface'

Wszystkie potrzebne pliki powinny znajdować się w tym samym katalogu co notatnik.
"""

import cv2
import dlib
import insightface

"""## Obraz testowy"""

img = cv2.imread("2_Demonstration_Demonstration_Or_Protest_2_58.jpg")
# przygotowanie obrazów: monochromatycznego i RGB (cv2.imread() zwraca obraz w formacie BGR - inna kolejność składowych)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cv2.imshow("img", img)

"""
## Kaskada Haara

Opracowano na podstawie: https://www.pyimagesearch.com/2021/04/05/opencv-face-detection-with-haar-cascades/
"""

# utworzenie i inicjalizacja detektora
haar_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# wywołanie detektora dla określonego obrazu (img_gray)
# wynikiem jest lista prostokątów w formacie [x, y, width, height]
haar_results = haar_detector.detectMultiScale(img_gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30),
                                              flags=cv2.CASCADE_SCALE_IMAGE)
print(len(haar_results))

# narysowanie wyników na kopii obrazu i wyświetlenie
img_haar = img.copy()
for (x, y, w, h) in haar_results:
    cv2.rectangle(img_haar, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.imshow("img_haar", img_haar)

"""
## Histogram zorientowanych gradientów (HOG) z maszyną wektorów nośnych (SVM)

Opracowano na podstawie: https://www.pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/
"""

# utworzenie i inicjalizacja detektora
hog_svm_detector = dlib.get_frontal_face_detector()

# wywołanie detektora dla określonego obrazu (img_rgb)
# wynikiem jest lista obiektów rectangle, zawierających współrzędne lewego górnego i prawego dolnego
# narożnika prostokąta
hog_svm_results = hog_svm_detector(img_rgb, 1)
print(len(hog_svm_results))

# narysowanie wyników na kopii obrazu i wyświetlenie
img_hog_svm = img.copy()
for rect in hog_svm_results:
    cv2.rectangle(img_hog_svm, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)
cv2.imshow("img_hog_svm", img_hog_svm)

"""
## Splotowa sieć neuronowa (CNN) z biblioteki dlib

Opracowano na podstawie: https://www.pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/
"""

# utworzenie i inicjalizacja detektora
cnn1_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

# wywołanie detektora dla określonego obrazu (img_rgb)
# wynikiem jest lista obiektów mmod_rectangle, zawierających m.in. pole rect ze współrzędnymi lewego górnego i
# prawego dolnego narożnika prostokąta
cnn1_results = cnn1_detector(img_rgb, 1)
print(len(cnn1_results))

# narysowanie wyników na kopii obrazu i wyświetlenie
img_cnn1 = img.copy()
for res in cnn1_results:
    rect = res.rect
    cv2.rectangle(img_cnn1, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)
cv2.imshow("img_cnn1", img_cnn1)

"""
## InsightFace - inna sieć neuronowa

Opracowano na podstawie: https://github.com/deepinsight/insightface/tree/master/python-package
"""

# utworzenie i inicjalizacja detektora
model_name = 'buffalo_s'  # model: small (_s), medium (_m) lub large (_l)
insf_detector = insightface.app.FaceAnalysis(name=model_name, root='insightface',
                                             allowed_modules=['detection'], providers=['CPUExecutionProvider'])
insf_detector.prepare(ctx_id=0, det_size=(1024, 1024))

# wywołanie detektora dla określonego obrazu
# wynikiem jest lista obiektów, zawierających m.in. pole bbox ze współrzędnymi lewego górnego i prawego dolnego
# narożnika prostokąta
insf_results = insf_detector.get(img)
print(len(insf_results))

# narysowanie wyników na kopii obrazu i wyświetlenie
img_insf = img.copy()
for res in insf_results:
    # współrzędne prostokątów sa zapisane jako liczby rzeczywiste - konwersja do liczb całkowitych
    rect = res.bbox.round().astype(int)
    cv2.rectangle(img_insf, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
cv2.imshow("img_insf", img_insf)

cv2.waitKey(0)  # zatrzymanie programu do czasu naciśnięcia dowolnego klawisza
cv2.destroyAllWindows()  # zamknięcie wszystkich okien
