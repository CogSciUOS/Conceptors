from matplotlib.pyplot import *
import copy


class Hierarchical:
    
    def __init__(self, RFC, M):
        
        self.RFC = RFC
        self.M   = M
            
        P_raw = np.zeros([self.RFC.K,self.RFC.n_patts])
        self.P = np.zeros_like(P_raw)
        
        for p in range(self.RFC.n_patts):
            P_raw[:,p] = np.mean(self.RFC.Z_list[p][:]**2,1) # mean over correct dimension? TR p. 164
            
        P_norm = np.sqrt(np.sum(P_raw**2,0))
        P_norm_mean = np.mean(P_norm)

        self.P = np.dot(P_raw,np.diag(1./P_norm))*P_norm_mean
    
    #@jit
    def run(self, patterns, pattTimesteps = 4000, plotRange = None, sigma = 0.99, drift = 0.01, gammaRate = 0.002, dcsv = 8, SigToNoise = 0.5):
        
        self.patterns      = patterns
        self.pattTimesteps = pattTimesteps if type(pattTimesteps) is list else [pattTimesteps for i in range(len(self.patterns))]
        self.n_patts       = len(self.patterns)
        self.plotRange     = plotRange if plotRange  is not None else self.pattTimesteps
        
        gamma = np.ones([self.M,self.n_patts])
        #gamma = (1./np.linalg.norm(gamma[0,:]))*gamma
        gamma = (1./np.sum(gamma[0,:])) * gamma
        
        tau = np.zeros([self.M])
        tau[1:] = 0.5
        
        self.deltaColl = []
        self.tauColl = []
        
        self.inp_colls    = []
        self.cl_inp_colls = []
        self.outp_colls   = []
        
        z       = np.zeros([self.M,self.RFC.K])
        y_bar   = np.zeros([self.M, self.RFC.n_ip_dim])
        vy_bar  = np.ones([self.M, self.RFC.n_ip_dim])
        delta   = np.ones([self.M])*0.5
        c       = np.ones([self.RFC.K])
        c_fin   = np.ones([self.M,self.RFC.K])
        c_aut   = np.ones([self.M,self.RFC.K])
        
        c_fin[:,:] = c = np.dot(self.P,gamma[0].T**2)/(np.dot(self.P,gamma[0].T**2) + self.RFC.alpha**-2)
        c_aut      = c_fin
        
        noiseLVL = np.sqrt(np.var(self.RFC.TrainOuts) / SigToNoise)
        
        """ Noised Run """
        for i,p in zip(range(self.n_patts), self.patterns):
            
            gammaColl = np.zeros((self.M, self.n_patts, self.pattTimesteps[i]))
            deltaColl = np.zeros((self.M, self.pattTimesteps[i]))
            tauColl = np.zeros_like(deltaColl)
            
            inp_coll = np.zeros([self.plotRange[i], self.RFC.n_ip_dim])
            cl_inp_coll = np.zeros([self.plotRange[i], self.RFC.n_ip_dim])
            outp_coll = np.zeros([self.plotRange[i], self.RFC.n_ip_dim])
            
            for t in range(self.pattTimesteps[i]):
                
                if type(p) == np.ndarray:
                    cl_inp = np.reshape(p[t], self.RFC.n_ip_dim)
                else:
                    cl_inp = np.reshape(p(t), self.RFC.n_ip_dim)
                    
                inp    = cl_inp + noiseLVL*np.random.randn(self.RFC.n_ip_dim) 

                for l in range(self.M):    

                    if (l == 0): y = inp                                   
                    u        = (1 - tau[l])*y + tau[l]*np.dot(self.RFC.D,z[l])
                    inaut    = np.dot(self.RFC.D,z[l]) #calculate after update of z[l]? TR p. 123
                    r        = np.tanh(np.dot(self.RFC.G,z[l]) + np.dot(self.RFC.W_in,u)+self.RFC.W_bias)
                    z[l]     = c_fin[l]*np.dot(self.RFC.F,r)  
                    y_pre    = y
                    y        = np.dot(self.RFC.W_out,r)
                    c_aut[l] = c_aut[l]+self.RFC.c_adapt_rate*(z[l]*z[l]-c_aut[l]*z[l]*z[l]-(self.RFC.alpha**-2)*c_aut[l])            
                    
                    y_bar[l] =  sigma*y_bar[l] + (1-sigma)*y                       #smoothing y
                    vy_bar[l]=  sigma*vy_bar[l] + (1-sigma)*(y - y_bar[l])**2
                    delta[l] =  np.max(sigma*delta[l] + (1-sigma)*((y_pre - inaut)**2)/vy_bar[l])         # smoothed error between y and self.RFC.Dz
                    # use vy_bar[l-1] instead? TR p. 123
                deltaColl[:,t] = delta          
                   
                for l in range(self.M-1):
                    tau[l+1] = 1/(1.+(delta[l+1]/delta[l])**dcsv)       
                
                tauColl[:,t] = tau        
                
                for l in reversed(range(self.M)):             #Calculate the conceptors vectors seperately because you need to loop top down.
                    w = np.dot(self.P,gamma[l].T**2)
                    if (l == (self.M - 1)):
                        c = w/(w + self.RFC.alpha**-2)
                        
                    if (l<(self. M- 1)):
                        c = (1 - tau[l + 1])*c_aut[l] + tau[l + 1]*c
                    c_fin[l] = c        
                
                for l in range(self.M):
                    w           = np.dot(self.P,gamma[l].T**2)
                    gamma_star  = gamma[l] + gammaRate*self.n_patts*(np.dot(np.dot(np.transpose(z[l]**2-w),self.P),np.diag(gamma[l]))+drift*(0.5 - gamma[l]))
                    gamma[l]    = np.exp(gamma_star)/np.sum(np.exp(gamma_star))  
                    
                gammaColl[:,:,t] = gamma  
        
                if (t > (self.pattTimesteps[i] - self.plotRange[i])):
                    inp_coll[t - (self.pattTimesteps[i] - self.plotRange[i])] = inp   
                    cl_inp_coll[t - (self.pattTimesteps[i] - self.plotRange[i])] = cl_inp                    
                    outp_coll[t - (self.pattTimesteps[i] - self.plotRange[i])] = y
            
            if i == 0:
                self.gammaColl = gammaColl
            else:
                self.gammaColl = np.append(self.gammaColl, gammaColl, axis = 2)
            self.deltaColl.append(deltaColl)
            self.tauColl.append(tauColl)
            self.inp_colls.append(copy.copy(inp_coll))
            self.outp_colls.append(copy.copy(outp_coll))
            self.cl_inp_colls.append(copy.copy(cl_inp_coll))
        
    def plot_tau(self):
        
        figure()
        xspace = [np.linspace(0,self.pattTimesteps[i],self.pattTimesteps[i]) for i in range(self.n_patts)]
        plot(xspace[1],self.tauColl[:][1,:])
        plot(xspace[2],self.tauColl[:][2,:])
        suptitle('Taus') 
        
    def plot_gamma(self, pltMeanGamma = False, songLengths = None):
        
        t_all = np.sum(self.pattTimesteps)
        xspace = np.linspace(0, t_all, t_all)
        
        if pltMeanGamma:
            if songLengths is None: raise ValueError('If pltMeanGamma is True, list with song lengths has to be passed to the function')
            for l in range(self.M):
                figure()
                idx = 0
                gammas = []
                for p in range(self.n_patts):
                    gammaSong = np.zeros((self.n_patts, songLengths[p]))
                    nSongs = int(self.pattTimesteps[p]/songLengths[p])
                    for s in range(nSongs):
                        gammaSong += self.gammaColl[l,:,idx:idx+songLengths[p]]
                        idx += songLengths[p]
                    gammas.append(gammaSong/nSongs)
                fullSongLengths = np.sum(songLengths)
                xspace_tmp = np.arange(0, fullSongLengths)
                gammaPlt = np.zeros((self.n_patts,fullSongLengths))
                t = 0                
                for g in gammas:
                    gammaPlt[:,t:t+g.shape[1]] = g
                    t += g.shape[1]
                fillStart = 0
                for p in range(self.n_patts):
                    gamma = plot(xspace_tmp, gammaPlt[p,:], label='Gamma of pattern '+str(p))
                    fill_between(xspace_tmp, 0, 1, where = (xspace_tmp >= fillStart) * (xspace_tmp < fillStart + songLengths[p]), facecolor = gamma[0].get_color(), alpha = 0.2, label = 'Pattern ' + str(p))
                    fillStart += songLengths[p]
                legend()
                suptitle('Gamma lvl' + str(l))
            ylim(0,1)
            show()
        else:
            for l in range(self.M):
                figure()
                idx = 0
                for p in range(self.n_patts):
                    realSongFull = np.zeros_like(xspace)
                    realSong = np.sum(self.patterns[p], axis = 1) != 0
                    realSongFull[idx:idx + len(realSong)] = realSong
                    idx += len(realSong)
                    gamma = plot(xspace ,self.gammaColl[l,p,:].T, label='Gamma of pattern '+str(p))
                    fill_between(xspace, 0, 1, where=realSongFull == 1, facecolor=gamma[0].get_color(), alpha=0.2, label = 'Pattern ' +str(p))
                legend()
                suptitle('Gamma lvl' + str(l))
            ylim(0,1)
            show()
        
    
    def checkPerformance(self):
        
        t_all = np.sum(self.pattTimesteps)
        xspace = np.linspace(0, t_all, t_all)
        performance = np.zeros((self.M,self.n_patts))
        for l in range(self.M):
            idx = 0
            for p in range(self.n_patts): 
                choice = np.argmax(np.squeeze(self.gammaColl[l,:,:]), axis = 0)
                realSongFull = np.zeros_like(xspace)
                realSong = np.sum(self.patterns[p], axis = 1) != 0
                realSongFull[idx:idx + len(realSong)] = realSong
                idx += len(realSong)
                performance[l,p] = np.mean(choice[realSongFull == 1] == p)
        
        return performance

        
    def plot_recall(self):
        
        figure() 
        for p in range(self.n_patts):
            subplot(self.n_patts,1,p+1)
            xspace = np.linspace(0,self.plotRange[p],self.plotRange[p])
            plot(xspace,self.cl_inp_colls[p], 'b') 
            plot(xspace,self.outp_colls[p], 'r')
            ylabel('syllable #')
        xlabel('t in steps')
        show()
    
    def plot_input(self):
        
        for p in range(self.n_patts):
            matshow((self.inp_colls[p]).T, cmap = 'jet', vmin = 0, vmax = 1, interpolation = 'nearest')
            xlabel('t in steps')
            ylabel('syllable #')
            title('Input over time')
            colorbar()