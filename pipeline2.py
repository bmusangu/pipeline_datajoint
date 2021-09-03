# Import the necessary packages
import scipy.io as spio
import datajoint as dj
import numpy as np 
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import random

plt.rcParams["figure.figsize"] = (8,5.5)


schema = dj.schema('bmusangu_pipeline', locals()) # create a schema

@schema
class Session(dj.Manual):
    definition = """
    # Session
    mouse_id:   int            # unique to a each mouse
    session_id: date           # session ID given by the date of the session
    """
    
@schema
class Stimulus(dj.Imported):     # subclass of session
    definition = """
    # Stimulus
    -> Session
    trial_id: int              # trial identified by trial number
    ---
    visori: int                # Trial value of grating drift direction
    viscon: int                # Trials value indicating the contrast of visual stimuli
    """
           
    def make(self, key):

        filename = 'data/AJ0{mouse_id}_{session_id}'.format(**key) # get the filename of the session you are interested in
        mat = spio.loadmat(filename, squeeze_me=True,struct_as_record=False) #load the data in .mat format
        data = mat[list(mat)[-1]] # unpack the dictionaries to select the specific data
        
        trial_id = 0
        for ori,con in zip(data.visOri, data.visCon):
            if con == 0.1:
                con = 0
            else:
                con = 1

            key['trial_id'] = trial_id
            key['visori'] = ori
            key['viscon'] = con
            trial_id += 1
            self.insert1(key) 
                
                
@schema
class Neuralactivity(dj.Imported):     # subclass of stimulus
    definition = """
    # Neural Activity
    -> Stimulus
    neuro_id: int        # unique
    ---
    activity: float      # electric activity of the neuron
    """
 
    def make(self, key):       

        filename = 'data/AJ0{mouse_id}_{session_id}'.format(**key) # get the filename of the session you are interested in
        mat = spio.loadmat(filename, squeeze_me=True,struct_as_record=False) #load the data in .mat format
        data = mat[list(mat)[-1]] # unpack the dictionaries to select the specific data

        activity_arr = data.deResp
        n_trials, n_neuron = activity_arr.shape 
        
        for neuro_id in range(0, n_neuron):
            key['neuro_id'] = neuro_id
            key['activity'] = activity_arr[int('{trial_id}'.format(**key)), neuro_id]
            self.insert1(key)
            
#         count = 0
#         if count <= n_neuron:
#             key['neuro_id'] = count
#             key['activity'] = activity_arr[int('{trial_id}'.format(**key)), count]
#             count += 1
#             self.insert1(key, skip_duplicates=True)
        
@schema
class ActivityStatistics(dj.Computed):
    definition = """
    # Activity Statistics
    -> Session
    neuro_id: int              # unique
    visori: int                # Trial value of grating drift direction
    viscon: int                # Trials value indicating the contrast of visual stimuli
    ---
    mean: float                # mean activity
    stdev: float               # standard deviation of activity
    max: float
    min: float
    """
    
    def make(self, key):
        # compute various statistics on activity
        
        data = (dj.U('neuro_id') & (Neuralactivity & 'mouse_id={mouse_id}'.format(**key) & 
                                    'session_id="{session_id}"'.format(**key))) * (dj.U('visori','viscon') & Stimulus)
        
        for stim in data:            
            
            activity = ((Neuralactivity & 'neuro_id = {n}'.format(n=stim['neuro_id']))*
                        (Stimulus & 'visori = {o}'.format(o=stim['visori']) & 'viscon = {c}'.format(c=stim['viscon']))
                        ).fetch('activity')  # fetch activity as NumPy array
            
            key['neuro_id'] = stim['neuro_id']
            key['visori'] = stim['visori']
            key['viscon'] = stim['viscon']
            key['mean'] = activity.mean()                # compute mean
            key['stdev'] = activity.std()                # compute standard deviation
            key['max'] = activity.max()                  # compute max
            key['min'] = activity.min()                  # compute min
            self.insert1(key)

@schema
class Parameters(dj.Computed):
    definition = """
    # Results
    -> ActivityStatistics   
    ---
    params: longblob            # set of parameters from computations
    act_mean_per_ori: longblob  # set of activity means per orientation for each neuron
    """
    
    def make(self, key):
        
        # get the number of neurons per_session
        n_neurons_sess = len(dj.U('neuro_id').aggr(Neuralactivity & 'mouse_id={mouse_id}'.format(**key) 
                         & 'session_id="{session_id}"'.format(**key), n="count(*)"))
        
        list_act_mean = {}
        model_list = ['m1', 'm2'] 

        for n_id in range(n_neurons_sess):
            
            for model in model_list:

                # fetch the neural activity and orietation 
                ori, act = ((Neuralactivity & 'mouse_id={mouse_id}'.format(**key) & 'session_id="{session_id}"'.format(**key)
                             & 'neuro_id={neuro_id}'.format(**key))
                            * (Stimulus & 'viscon={viscon}'.format(**key))).fetch('visori', 'activity')

                visori_con = pd.DataFrame(np.vstack((ori, act)).T)
                visori_con.columns = ['ori', 'act']
                means = pd.DataFrame(visori_con.groupby('ori')['act'].mean()).reset_index()
                act_mean = means.act
                ori_mean = means.ori*(np.pi/180.0)
                ori = ori*(np.pi/180.0)
                sigma = visori_con.act.std()

                if model == 'm2':            
                    # initialize with random starting points
                    x0_arr = [0.01 + 0.2 * np.random.randint(1, 5, 5) for x in range(9)] 
                    a = min(act_mean)
                    b = (max(act_mean)-min(act_mean))/2.0
                    b2 = max(act_mean)/2.0
                    c = 90*(np.pi/180.0)
                    theta_p = max(act_mean)*(np.pi/180.0)
                    x0_arr.append(np.array([a, b, c, theta_p, b2])) # initial guesses
                else:
                    # initialize with random starting points
                    x0_arr = [0.01 + 0.2 * np.random.randint(1, 4, 4) for x in range(9)]
                    a = min(act_mean)
                    b = (max(act_mean)-min(act_mean))/2.0
                    c = 90 * (np.pi/180.0)
                    theta_p = max(act_mean)*(np.pi/180.0)
                    x0_arr.append(np.array([a, b, c, theta_p])) # initial guesses 

                def L_fun(x):
                    if model == 'm2':
                        activity = act_mean
                        J = np.mean(((activity - model_fun2(x, ori_mean))/sigma)**2.0)
                    else:
                        activity = act
                        J = np.mean((activity - model_fun1(x, ori))**2.0)
                    return J           

                params = {}
                c = 0
                for x0 in x0_arr:

                    res = minimize(L_fun, x0, options={'disp':False})
                    params[c] = (res.x)
                    c += 1

                j_list = {}
                for x0 in [*params]:
                    J = L_fun(params[x0])
                    j_list[x0] = J
                lowest_j = min(j_list, key=j_list.get)
                
                key['params'] = params[lowest_j]
                key['act_mean_per_ori'] = list(act_mean)
                self.insert1(key)

            

#--------------------------------------------------------Models-------------------------------------------------------

# model function 1
def model_fun1(x, theta): 
    return x[0] + (x[1] * np.exp(x[2] * np.cos(2.0 * (theta - x[3]))))

# model function 2    
def model_fun2(x, theta):    
    return x[0] + (x[1] * np.exp(x[2] * np.cos(theta - x[3]))) + (x[4] * np.exp(-x[2] * np.cos(theta - x[3])))

#--------------------------------------------------------Helper Methods-------------------------------------------------------

def pick_model(x, phi, model): 
    if model == 'm2':
        model = model_fun2(x, phi)
    else:
        model = model_fun1(x, phi)
    return model

# auto populate
def populate_new_data():
    
    Stimulus.populate(display_progress=True)
    Neuralactivity.populate(display_progress=True)
    ActivityStatistics.populate(display_progress=True)
    Parameters.populate(display_progress=True)
    
    print("All tables are populated.")
    
# plot of tuning curves
def tuning_curve_plots(mouse_id, session_id, viscon, model):
    
    list_keys = (Parameters & f'mouse_id = {mouse_id}' & f'session_id = {session_id}' & f'viscon = {viscon}').fetch('KEY')
    rand_keys = random.sample(list_keys, 16)
    ori_mean = (dj.U('visori') & Parameters.proj())

    for key in rand_keys:

        act_mean = '{act_mean}'.format(**key)
        x0 = '{params}'.format(**key)

        # plot 16 random fitted tuning curves 
        fig, axs = plt.subplots(4, 4, figsize=(15,10))
        fig.suptitle("16 Randomly Fitted Tuning Curves", y=0.94, fontsize='xx-large')
        best_param_keys = [*best_fit_params]
        count = 0
        for i in range(4):
            for j in range(4):

                axs[i, j].plot(ori_mean, act_mean)
                phi = np.linspace(0,2*np.pi, 100)              
                y = pick_model(x0, phi, model)
                axs[i, j].plot(phi, y)
                axs[i, j].set_title(f'Neuron {rand_nid[count]}')
                count += 1

    for ax in axs.flat:
        ax.label_outer()
        
    # set labels
    plt.setp(axs[-1, :], xlabel='Orientation')
    plt.setp(axs[:, 0], ylabel='$\Delta$F/F')
    

#--------------------------------------------------------Random Methods-------------------------------------------------------


def neuroAnalysis(m_id, sess_id, conid, model):
    
    """
    Inputs:
    m_id ---> mouse_id as int
    sess_is ---> session_id as int: e.g. 190902 or str: '2019-09-02'
    conid ---> contrast as int: high = 1 or low = 0
    model ---> str: model with 4 parameters = 'm1' and it is the default model
                    model with 5 parameters = 'm2'
    --------------------------------------------------------------------------
    Output:
    plots of fitted tuning curves
    """

    # get the number of neurons per_session
    n_neurons_sess = len(dj.U('neuro_id').aggr(Neuralactivity & f'mouse_id={m_id}' & f'session_id={sess_id}', n="count(*)"))
    
    # generate randon neuroId
    rand_nid = np.random.randint(0, n_neurons_sess, 16)
    
    best_fit_params = {} # best fit params for each neuron
    list_act_mean = {}
    
    for n_id in rand_nid:
        
        # fetch the neural activity and orietation 
        ori, act = ((Neuralactivity & f'mouse_id={m_id}' & f'session_id={sess_id}' & f'neuro_id={n_id}')
                    * (Stimulus & f'viscon={conid}')).fetch('visori', 'activity')

        visori_con = pd.DataFrame(np.vstack((ori, act)).T)
        visori_con.columns = ['ori', 'act']
        means = pd.DataFrame(visori_con.groupby('ori')['act'].mean()).reset_index()
        act_mean = means.act
        ori_mean = means.ori*(np.pi/180.0)
        ori = ori*(np.pi/180.0)
        sigma = visori_con.act.std()
        
        if model == 'm2':            
            # initialize with random starting points
            x0_arr = [0.01 + 0.2 * np.random.randint(1, 5, 5) for x in range(9)] 
            a = min(act_mean)
            b = (max(act_mean)-min(act_mean))/2.0
            b2 = max(act_mean)/2.0
            c = 90*(np.pi/180.0)
            theta_p = max(act_mean)*(np.pi/180.0)
            x0_arr.append(np.array([a, b, c, theta_p, b2])) # initial guesses
        else:
            # initialize with random starting points
            x0_arr = [0.01 + 0.2 * np.random.randint(1, 4, 4) for x in range(9)]
            a = min(act_mean)
            b = (max(act_mean)-min(act_mean))/2.0
            c = 90 * (np.pi/180.0)
            theta_p = max(act_mean)*(np.pi/180.0)
            x0_arr.append(np.array([a, b, c, theta_p])) # initial guesses 
        
        def L_fun(x):
            if model == 'm2':
                activity = act_mean
                J = np.mean(((activity - model_fun2(x, ori_mean))/sigma)**2.0)
            else:
                activity = act
                J = np.mean((activity - model_fun1(x, ori))**2.0)
            return J           

        params = {}
        c = 0
        for x0 in x0_arr:
            
            res = minimize(L_fun, x0, options={'disp':False})
            params[c] = (res.x)
            c += 1
            
        j_list = {}
        for x0 in [*params]:
            J = L_fun(params[x0])
            j_list[x0] = J
        lowest_j = min(j_list, key=j_list.get)
        best_fit_params[n_id] = params[lowest_j]
        list_act_mean[n_id] = act_mean
                                       
    # plot 16 random fitted tuning curves 
    fig, axs = plt.subplots(4, 4, figsize=(15,10))
    fig.suptitle("16 Randomly Fitted Tuning Curves", y=0.94, fontsize='xx-large')
    best_param_keys = [*best_fit_params]
    count = 0
    for i in range(4):
        for j in range(4):
        
            axs[i, j].plot(ori_mean, list_act_mean[best_param_keys[count]])
            phi = np.linspace(0,2*np.pi, 100)
            x = best_fit_params[best_param_keys[count]]              
            y = pick_model(x, phi, model)
            axs[i, j].plot(phi, y)
            axs[i, j].set_title(f'Neuron {rand_nid[count]}')
            count += 1
            
    for ax in axs.flat:
        ax.label_outer()
        
    # set labels
    plt.setp(axs[-1, :], xlabel='Orientation')
    plt.setp(axs[:, 0], ylabel='$\Delta$F/F')   
    
        
     
        
        
    
    
    
    
    
    
    
    
    
    
    
    