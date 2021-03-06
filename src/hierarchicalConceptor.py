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
                inp    = inp - min(inp)
                inp    = inp / max(inp)

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

        self.class_predictions = self.gammaColl.argmax(axis = 1)

    def plot_tau(self):

        figure()
        xspace = [np.linspace(0,self.pattTimesteps[i],self.pattTimesteps[i]) for i in range(self.n_patts)]
        plot(xspace[1],self.tauColl[:][1,:])
        plot(xspace[2],self.tauColl[:][2,:])
        suptitle('Taus') 
    
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

    def plot_gamma(self, songLenghts = None):

        t_all = np.sum(self.pattTimesteps)
        xspace = np.arange(t_all)

        # make a figure for every HFC level
        for l in range(self.M):

            # plot gamma and play-area for all patterns
            for p_idx, p in enumerate(self.patterns):

                # calculate start and stop idxs for this song in whole classification
                start_idx = np.sum(self.pattTimesteps[0:p_idx], dtype = np.int)
                end_idx = np.sum(self.pattTimesteps[0:p_idx + 1], dtype = np.int)

                # plot gamma values for this song
                gamma_plot = plot(xspace, self.gammaColl[l, p_idx, :].T,
                    label = 'Gamma of song {}'.format(p_idx))

                # show areas where the song was played
                pattern_not_empty = p.any(axis = 1)
                fill_between(np.arange(start_idx, end_idx), -0.2, 0,
                    where = pattern_not_empty,
                    facecolor = gamma_plot[0].get_color(),
                    alpha = 0.2,
                    label = 'Song {}'.format(p_idx),
                    linewidth = 0,
                    )

                # plot lines after every single song iteration
                if songLenghts:
                    for i in range(start_idx, end_idx):
                        if (i-start_idx) % songLenghts[p_idx] == 0:
                            axvline(i, ymin=0, ymax=1, color = 'black', alpha = 0.2)

                # plot class predictions
                this_pattern_prediction = self.class_predictions[l] == p_idx
                fill_between(xspace, 0, 1,
                    where = this_pattern_prediction,
                    facecolor = gamma_plot[0].get_color(),
                    alpha = 0.2,
                    linewidth = 0,
                    )

                # seperator for target and predictions
                axhline(0, color = 'black')

            # dummy plot for label for song border lines
            plot([], [], color = 'black', alpha = 0.2, label = "Song borders (in target)")

            # set y axis for gamma levels
            ylim(-0.2, 1)
            xlim(0, xspace[-1])
            yticks(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
            ylabel('Gamma')

            # make legend (must be here, because the data belongs to standard y-axis)
            legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', borderaxespad=0.0, ncol=2)

            # switch y axis and set song labels
            twinx()
            ylim(-0.2, 1)
            yticks([-0.1, 0.5], ["Target", "Predicted"])
            ylabel('Song')

            # tight layout removes layouting issues with the twinx y-axis
            #tight_layout()

            # create dynamic offset for the legend depending on number of patterns
            legend_offset = 0.2 + self.n_patts * 0.05
            gcf().subplots_adjust(bottom=legend_offset)

        # show all figures for all hfc levels
        show()

    def plot_recall(self):

        figure()
        for p in range(self.n_patts):
            
            subplot(self.n_patts, 1, p+1)

            # convert categorical data back to syllable id or 0 if no syllable present
            inp = [cat.argmax()+1 if cat.any() else 0 for cat in self.cl_inp_colls[p]]
            out = [cat.argmax()+1 if cat.any() else 0 for cat in self.outp_colls[p]]

            xspace = np.arange(self.plotRange[p])
            plot(xspace, inp, 'g', label = 'Target')
            plot(xspace, out, 'b', label = 'Output')
            title('Recall pattern {}'.format(p+1))
            legend()
        tight_layout()
        show()
    
    def plot_input(self):
        
        for p in range(self.n_patts):
            matshow((self.inp_colls[p]).T, cmap = 'jet', vmin = 0, vmax = 1, interpolation = 'nearest')
            xlabel('t in steps')
            ylabel('syllable #')
            title('Input over time')
            colorbar()