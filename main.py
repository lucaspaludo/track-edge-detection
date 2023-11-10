import cv2
import numpy as np

#abre o vídeo
captura = cv2.VideoCapture("video.mp4")

#abre a imagem
imagem = cv2.imread("carro.png")

#formata a imagem
imagem_formatada = cv2.resize(imagem, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)

#posição inicial do carro
x, y = 650, 580


while(1):

    #inicia a leitura do vídeo
    ret, frame = captura.read()

    #se conseguir abrir o vídeo
    if ret:     

        #obtém altura e largura da imagem   
        altura, largura, _ = imagem_formatada.shape

        #obtém metade da largura da imagem
        metade_largura = largura/2

        
        #PROCESSAMENTO DO VÍDEO
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        suave = cv2.GaussianBlur(gray, (7, 7), 0)
        canny = cv2.Canny(suave, 20, 120)

        linha_analise = canny[600,:]


        #Sobreposição da imagem no vídeo
        frame[y:y+altura, x:x+largura] = imagem_formatada
        
          
        pixel_borda_direita = linha_analise[x+int(metade_largura)+200] - linha_analise[x]
        pixel_borda_esquerda = linha_analise[x-int(metade_largura)-100] - linha_analise[x]
        
        if pixel_borda_direita > pixel_borda_esquerda:
            x-=10   

        if pixel_borda_esquerda > pixel_borda_direita:
            x+=10

        #cv2.imshow('Video Sobreposto', frame)
        cv2.imshow('Identificação de borda', frame)
  
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


