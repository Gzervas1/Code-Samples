#from groupC_functions import *
from functions import *

def groupX_numerical(mockdata, signalmaps):
    
    def groupX_chi_squared(c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13):
        """
        Function that calculates chi squared for a set of best fit parameters
        Input: Data set, theoretical signal maps and best fit normalizations
        Output: Chi-squared
        """

        s_ph_i  = signalmaps[0,:,:,:]
        s_WT_i  = signalmaps[1,:,:,:]
        s_ZT_i  = signalmaps[2,:,:,:]
        s_bb_i  = signalmaps[3,:,:,:]
        s_hh_i  = signalmaps[4,:,:,:]
        s_tt_i  = signalmaps[5,:,:,:]
        b_i     = signalmaps[6,:,:,:]
        gce1_i  = signalmaps[7,:,:,:]
        gce2_i  = signalmaps[8,:,:,:]
        gce3_i  = signalmaps[9,:,:,:]
        gce4_i  = signalmaps[10,:,:,:]
        ic_i    = signalmaps[11,:,:,:]
        fb_i    = signalmaps[12,:,:,:]

        numerator_sqrt_i = mockdata-c1*s_ph_i-c2*s_WT_i-c3*s_ZT_i-c4*s_bb_i-c5*s_hh_i-c6*s_tt_i \
            -c7*b_i-c8*gce1_i-c9*gce2_i-c10*gce3_i-c11*gce4_i-c12*ic_i-c13*fb_i
        numerator_i = numerator_sqrt_i**2
        chi_squared = np.sum(np.where(mockdata!=0,numerator_i/(mockdata+1.e-100),0))

        return chi_squared
    
    minima = Minuit(groupX_chi_squared,c1=1.,c2=1.,c3=1.,c4=1.,c5=1.,c6=1.,c7=1.,c8=1.,
                    c9=1.,c10=1.,c11=1.,c12=1.,c13=1.)
    minima.errordef = 1 # 0.5 for loglikelihood and change to 1 for least squares!!!
    minima.limits = [(0,100),(0,100),(0,100),(0,100),(0,100),(0,100),
                     (0,100),(0,100),(0,100),(0,100),(0,100),(0,100),(0,100)]
    minima.fixed['c1']  = False 
    minima.fixed['c2']  = False 
    minima.fixed['c3']  = False 
    minima.fixed['c4']  = False 
    minima.fixed['c5']  = False 
    minima.fixed['c6']  = False 
    minima.fixed['c7']  = False 
    minima.fixed['c8']  = False 
    minima.fixed['c9']  = False 
    minima.fixed['c10'] = False 
    minima.fixed['c11'] = False 
    minima.fixed['c12'] = False 
    minima.fixed['c13'] = False 

    minima.params

    migrad = minima.migrad()
    #minos = minima.minos()
    
    return np.array(minima.values),minima.fval


def groupX_numerical_mass_scan(mockdata, signalmaps, masslist):
    index = np.arange(np.alen(signalmaps))
    indexnew = index[::100]
    signalmapsnew = signalmaps[indexnew]
    masslistnew = masslist[indexnew]
    numerical_result = list(map(lambda x: groupX_numerical(mockdata,x),signalmapsnew))
    chi2min = 1e10
    for i in np.arange(np.alen(indexnew)):
        chi2 = numerical_result[i][1]
        if chi2<chi2min:
            ibest = i
            chi2min = chi2
    indexbest = indexnew[ibest]

    if indexbest>=100:
        indexnew = index[indexbest-100:indexbest+101:10]
    else:
        indexnew = index[0:indexbest+101:10]
    signalmapsnew = signalmaps[indexnew]
    masslistnew = masslist[indexnew]
    numerical_result = list(map(lambda x: groupX_numerical(mockdata,x),signalmapsnew))
    chi2min = 1e10
    for i in np.arange(np.alen(indexnew)):
        chi2 = numerical_result[i][1]
        if chi2<chi2min:
            ibest = i
            chi2min = chi2
    indexbest = indexnew[ibest]

    indexnew = index[indexbest-10:indexbest+11]
    signalmapsnew = signalmaps[indexnew]
    masslistnew = masslist[indexnew]
    numerical_result = list(map(lambda x: groupX_numerical(mockdata,x),signalmapsnew))
    chi2min = 1e10
    for i in np.arange(np.alen(indexnew)):
        chi2 = numerical_result[i][1]
        if chi2<chi2min:
            ibest = i
            chi2min = chi2
    indexbest = indexnew[ibest]

    return indexbest,numerical_result[ibest]



def groupX_iterations(iteration, template_id=1):

    #psi = groupA_psigrid()
    #Jmap = groupA_J(psi,flavour='NFW',nPts=2000)
    data_template = np.load('dataX/final_counts_map_%s.npy'%template_id)
    masslist = np.linspace(2400,3200,801)*GeV
    sv_ref = 1.e-25*cm**3/s
    signalmaps = np.load('dataX/theoretical_templates.npy')
    mask_2d = np.loadtxt('dataB/0414/mask_0414.txt')
    data_template_new = data_template*mask_2d
    mockdata = np.random.poisson(data_template_new*np.ones((iteration,1,1,1)))
    for i in np.arange(iteration):
        mockdatanew = mockdata[i]
        numerical_result = groupX_numerical_mass_scan(mockdatanew,signalmaps,masslist)
        indexbest = numerical_result[0]
        bestfit = np.concatenate((np.array([masslist[indexbest]]),numerical_result[1][0]))
        bestfit[1:7] = bestfit[1:7]*sv_ref/(cm**3/s)
        if i==0:
            bestfitall = bestfit
        else:
            bestfitall = np.vstack((bestfitall,bestfit))
        print('Iteration %s finished:'%(i+1))
        print(bestfit)
    return bestfitall


def groupX_iterations_binbybin(iteration, template_id=1):

    Jmap = groupA_J(groupA_psigrid(),flavour='NFW',nPts=10000)
    data_template = np.load('dataX/final_counts_map_%s.npy'%template_id)
    m_ref  = TeV 
    sv_ref = 1.e-25*cm**3/s
    N_ref  = 1.
    if template_id==2 or template_id==4 or template_id==6 or template_id==8:
        signalmaps = groupC_loaddata(None,Jmap,bin_by_bin=True,m_ref=m_ref,sv_ref=sv_ref,N_ref=N_ref,fb='max')
    else:
        signalmaps = groupC_loaddata(None,Jmap,bin_by_bin=True,m_ref=m_ref,sv_ref=sv_ref,N_ref=N_ref)


    energy_central = np.loadtxt('dataB/0331/energies_0331.txt')*TeV
    energy_H = np.loadtxt('dataB/0407/energiesH_0407.txt')*TeV
    energy_L = np.loadtxt('dataB/0407/energiesL_0407.txt')*TeV
    mask_2d = np.loadtxt('dataB/0414/mask_0414.txt')
    bestfit = np.empty((np.alen(energy_central),iteration,len(signalmaps)))
    for i in np.arange(np.alen(energy_central)):
        data_template_new = data_template[i,:,:]*mask_2d
        signalmapsnew = signalmaps[:,i,:,:]
        mockdata = np.random.poisson(data_template_new*np.ones((iteration,1,1)))
        signalmapsnew = signalmapsnew*np.ones((iteration,1,1,1))
        bestfit[i] = np.array(list(map(lambda x,y: groupC_numerical(x,y),mockdata,signalmapsnew)))
        bestfit[i,:,0] *= sv_ref*N_ref/m_ref**2/(energy_H[i]-energy_L[i])
        print('Finished energy bin %s: %.1f GeV'%(i+1,energy_central[i]/GeV))
        print('Median best fit: %.3e cm^3/s/TeV\n'%(np.percentile(bestfit[i,:,0],50)*energy_central[i]**2/(cm**3/s/TeV)))
    return bestfit


def groupX_create_data_template(template_id):
    if template_id==1:
        mass = 2.7*TeV
        profile = 'NFW'
        fb = 'min'
    elif template_id==2:
        mass = 2.7*TeV
        profile = 'NFW'
        fb = 'max'
    elif template_id==3:
        mass = 2.7*TeV
        profile = 'CORED'
        fb = 'min'
    elif template_id==4:
        mass = 2.7*TeV
        profile = 'CORED'
        fb = 'max'
    elif template_id==5:
        mass = 3.*TeV
        profile = 'NFW'
        fb = 'min'
    elif template_id==6:
        mass = 3.*TeV
        profile = 'NFW'
        fb = 'max'
    elif template_id==7:
        mass = 3.*TeV
        profile = 'CORED'
        fb = 'min'
    elif template_id==8:
        mass = 3.*TeV
        profile = 'CORED'
        fb = 'max'

    template = groupB_total_counts_map(mass,profile,fb)
    np.save('dataX/final_counts_map_%s.npy'%template_id,template)


#def groupX_create_theoretical_template():
#    masslist = np.linspace(2500,3200,701)*GeV
#    J = groupA_J(groupA_psigrid(),'NFW',10000)
#    template = np.array(list(map(lambda x: groupC_loaddata(x,J),masslist)))
#    np.save('dataX/theoretical_templates.npy',template)
    

def groupX_create_theoretical_template():
    masslist = np.linspace(2500,3200,701)*GeV
    J = groupA_J(groupA_psigrid(),'NFW',10000)
    template = np.empty((701,13,47,20,20))
    for i in np.arange(np.alen(masslist)):
        mass = masslist[i]
        template[i] = groupC_loaddata(mass,J)
        print('%d %f finished'%(i,mass))
    np.save('dataX/theoretical_templates.npy',template)

        
def groupX_create_theoretical_template_2():
    masslist = np.linspace(2400,2499,100)*GeV
    J = groupA_J(groupA_psigrid(),'NFW',10000)
    template = np.empty((100,13,47,20,20))
    for i in np.arange(np.alen(masslist)):
        mass = masslist[i]
        template[i] = groupC_loaddata(mass,J)
        print('%d %f finished'%(i,mass))
    np.save('dataX/theoretical_templates_2.npy',template)

        

