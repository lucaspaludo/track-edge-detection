import cv2

def redim(img, largura):
    alt = int(img.shape[0]/img.shape[1]*largura)
    img = cv2.resize(img, (largura, alt), interpolation=cv2.INTER_AREA)
    return img

df = cv2.CascadeClassifier('cars.xml')

#abre o vídeo
captura = cv2.VideoCapture("video.mp4")

#abre a imagem
imagem = cv2.imread("carro.png")

#formata a imagem
imagem_formatada = cv2.resize(imagem, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)

#posição inicial do carro
x, y = 650, 580

while True:

    (sucesso, frame) = captura.read()
    if not sucesso:
        break

    #obtém altura e largura da imagem   
    altura, largura, _ = imagem_formatada.shape

    #obtém metade da largura da imagem
    metade_largura = largura/2

    
    
    #Sobreposição da imagem no vídeo
    frame[y:y+altura, x:x+largura] = imagem_formatada

    #frame = redim(frame, 500)
    frame_pb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    suave = cv2.GaussianBlur(frame_pb, (7, 7), 0)
    canny = cv2.Canny(suave, 20, 120)

    linha_analise = canny[600,:]
    
    
    faces = df.detectMultiScale(frame_pb, scaleFactor=1.1, minNeighbors=4, minSize=(70, 70), flags=cv2.CASCADE_SCALE_IMAGE)

    frame_temp = frame.copy()

    pixel_borda_direita = linha_analise[x+int(metade_largura)+150] - linha_analise[x]
    pixel_borda_esquerda = linha_analise[x-int(metade_largura)-20] - linha_analise[x]
      
    if pixel_borda_direita > pixel_borda_esquerda:
        x-=10   

    if pixel_borda_esquerda > pixel_borda_direita:
        x+=10

    for (a, b, lar, alt) in faces:
        cv2.rectangle(frame_temp, (a, b), (a+lar, b+alt), (0, 255, 255), 2)

    final = cv2.resize(redim(frame_temp, 640), (1080, 720), interpolation=cv2.INTER_AREA)

    cv2.imshow('Detector de faces', final)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break