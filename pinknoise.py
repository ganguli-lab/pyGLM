import numpy as np
import matplotlib.pyplot as plt

n = 100
t = 500
dt = 0.1

noise = np.random.randn(t,n,n)

tc = np.arange(t).reshape(( -1,1,1 )) * dt
xc = np.arange(n).reshape(( 1,-1,1 ))
yc = np.arange(n).reshape(( 1,1,-1 ))

tc -= np.mean(tc)
xc -= np.mean(xc)
yc -= np.mean(yc)

radii = np.sqrt( tc**2 + xc**2 + yc**2 )

# normalize
noise /= np.fft.fftshift(radii)

stim = np.fft.ifftn(noise)

#def one_over_f(f, knee, alpha):
    #desc = np.ones_like(f)
    #desc[f<KNEE] = np.abs((f[f<KNEE]/KNEE)**(-alpha))
    #desc[0] = 1
    #return desc

#white_noise_sigma =  3 #mK * sqrt(s)

#SFREQ = 2 #Hz
#KNEE = 5.0 / 1e3 #Hz
#ALPHA = .7
#N = SFREQ * 3600 * 2 # 4 hours

##generate white noise in time domain
#wn=np.random.normal(0.,white_noise_sigma*np.sqrt(SFREQ),N)

##shaping in freq domain
#s = np.fft.rfft(wn)
#f = np.fft.fftfreq(N, d=1./SFREQ)[:len(s)]
#f[-1]=np.abs(f[-1])
#fft_sim = s * one_over_f(f, KNEE, ALPHA)
#T_sim = np.fft.irfft(fft_sim)

##PSD - 1 hour window
#NFFT = int(SFREQ*60*60*1)
#s_sim, f_sim  = mlab.psd(T_sim, NFFT=NFFT, Fs=SFREQ, scale_by_freq=True)

##plot
#plt.figure()
#plt.plot(f_sim, np.sqrt(s_sim), label='sim')
#plt.loglog(f_sim, one_over_f(f_sim, KNEE, ALPHA) * white_noise_sigma*1e3*np.sqrt(2), 'r',label='noise model')
#plt.vlines(KNEE,*plt.ylim())
#plt.grid(); plt.xlabel('Freq'); plt.title('Amplitude spectrum'); plt.legend()
