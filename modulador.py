import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.fftpack import fft
import suaBibSignal
from funcoes_LPF import *
import matplotlib.pyplot as plt

def main():
    print("Comecou")
    sinal = suaBibSignal.signalMeu()

    #Passo1 - Leitura
    print("Lendo")
    leitura,f = sf.read("queen.wav", dtype="float64")
    print("Lido")
    leitura = leitura[:220500]
    fs = 44100

    # #Passo2 - Eliminando Freqs
    k = 1/max(abs(leitura))
    k_leitura = k*leitura
    freq_elim = 2500
    lpf = LPF(leitura, freq_elim, fs)
    sinal.plotFFT(lpf, fs)
    plt.title("Frequências Eliminadas")
    plt.show()

    #Passo3 - Verificando Audio
    sd.play(lpf, fs)
    sd.wait()
    
    #Passo 4 - Modulando
    #Sinal Modulado x Tempo
    x1, y1 = sinal.generateSin(1.3e4, 1, 5, fs)
    modulado = lpf*y1

    plt.plot(x1, modulado)
    plt.title("Modulado")
    plt.show()
    

    # #Sinal Modulado x Freq
    x2,y2 = sinal.calcFFT(modulado, fs)
    plt.plot(x2,y2)
    plt.show()

    sd.play(modulado, fs)
    sd.wait()

    # #Passo 5
    plt.title("Normalizado")
    sinal.plotFFT(modulado*y1, fs)
    plt.show()


    lpf = LPF(modulado*y1, freq_elim, fs)
    sinal.plotFFT(lpf, fs)
    plt.show()
    
    sd.play(lpf, fs)
    sd.wait()

if __name__ == "__main__":
    main()










