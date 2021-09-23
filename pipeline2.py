# Import the necessary packages
import scipy.io as spio
import datajoint as dj
import numpy as np 
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import itertools

# constants
MAX_BATCH_INSERT_SIZE = 10000   # maximum number of entries to insert at once
DATA_FOLDER = 'data'            # folder where data files are expected to reside

plt.rcParams["figure.figsize"] = (8,5.5)


schema = dj.schema('bmusangu_pipeline', locals()) # create a schema

@schema
class Session(dj.Manual):
    definition = """
    # Session
    mouse_id:   int            # unique to a each mouse
    session_date: date         # session date given by the date of the session
    """
    
@schema
class Stimulus(dj.Imported):     # subclass of session
    definition = """
    # Stimulus
    -> Session
    trial_id: int              # trial identified by trial number
    ---
    visori: int                # Trial value of grating drift direction
    viscon: int                # Trial value indicating the contrast of visual stimuli
    """
           
    def make(self, key):
        # load raw data
        filename = '{data_folder}/AJ0{mouse_id}_{session_date}'.format(
            data_folder=DATA_FOLDER, **key) # get the filename of the session you are interested in
        mat = spio.loadmat(filename, squeeze_me=True,struct_as_record=False) #load the data in .mat format
        data = mat[list(mat)[-1]] # unpack the dictionaries to select the specific data

        mouse_id = key['mouse_id']
        session_date = key['session_date']
        n_trials, n_neurons = data.deResp.shape
        # batch insert stimulus data into Stimulus table
        stimdata = [{
            'mouse_id': mouse_id,
            'session_date': session_date,
            'trial_id': trial_id,
            'visori': data.visOri[trial_id],
            'viscon': 0 if data.visCon[trial_id] == 0.1 else 1
        } for trial_id in range(n_trials)]
        self.insert(stimdata)
        
        # batch insert neural data into Neuralactivity table
        neurdata = [{
            'mouse_id': mouse_id,
            'session_date': session_date,
            'trial_id': trial_id,
            'neuro_id': neuro_id,
            'activity': data.deResp[trial_id, neuro_id]
        } for trial_id in range(n_trials) for neuro_id in range(n_neurons)]
        # limit number of simultaneously added entries. Not doing so seems to
        # lead to database connection issues (LostConnectionError).
        n_neurdata = len(neurdata)
        for batch_i in range(0, n_neurdata, MAX_BATCH_INSERT_SIZE):
            Neuralactivity.insert(
                neurdata[batch_i:min(batch_i + MAX_BATCH_INSERT_SIZE,
                                     n_neurdata)])
                
@schema
class Neuralactivity(dj.Manual):     # subclass of stimulus
    definition = """
    # Neural Activity
    -> Stimulus
    neuro_id: int        # unique
    ---
    activity: float      # electric activity of the neuron
    """

        
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
        # compute various statistics on neural activity

        # find unique visori/viscon combinations for each neuron in session.
        # In the below, we first restrict Neuralactivity to the entries
        # corresponding to the session of interest and then perform the join
        # with Stimulus. This should be faster than first performing the join
        # and then subselecting entries relevant for the current session.
        uniquestims = dj.U('neuro_id', 'visori', 'viscon') & \
            (Stimulus * (Neuralactivity
                         & 'mouse_id={mouse_id}'.format(**key)
                         & 'session_date="{session_date}"'.format(**key)))

        for stim in uniquestims:

            activity = (
                (Neuralactivity & 'neuro_id = {}'.format(stim['neuro_id'])) * 
                (Stimulus & 'visori = {}'.format(stim['visori']) 
                          & 'viscon = {}'.format(stim['viscon'])) 
                ).fetch('activity')  # fetch activity as NumPy array

            key['neuro_id'] = stim['neuro_id']
            key['visori'] = stim['visori']
            key['viscon'] = stim['viscon']
            key['mean'] = activity.mean()                # compute mean
            key['stdev'] = activity.std()                # compute std
            key['max'] = activity.max()                  # compute max
            key['min'] = activity.min()                  # compute min
            self.insert1(key)

@schema
class TuningCurveFits(dj.Computed):
    definition = """
    # Results
    -> Session 
    neuro_id: int        # unique
    viscon:   int        # Trials value indicating the contrast of visual stimuli
    model_id: varchar(5) # Type of model being fitted
    ---
    params: longblob            # set of parameters from computations
    act_mean_per_ori: longblob  # set of activity means per orientation for each neuron
    """
    
    def make(self, key):
        
        # get the number of neurons per_session
        n_neurons_sess = len(dj.U('neuro_id') & 
            (Neuralactivity & 'mouse_id={mouse_id}'.format(**key) 
                & 'session_date="{session_date}"'.format(**key)))

        model_list = ['m4p', 'm5p']                    # list of models
        uniquecons = (dj.U('viscon') & ActivityStatistics)  

        for model in model_list:
            for con_id in uniquecons:
                for n_id in range(n_neurons_sess):
                    # fetch the neural activity and orietation 
                    ori, act = ((Neuralactivity 
                            & 'mouse_id={mouse_id}'.format(**key) 
                            & 'session_date="{session_date}"'.format(**key) 
                            & f'neuro_id={n_id}') * 
                            (Stimulus & f'viscon={con_id["viscon"]}') 
                        ).fetch('visori', 'activity')

                    vis_ori_con = pd.DataFrame(data={'ori': ori, 'act': act})
                    mean_act_by_ori = pd.DataFrame(vis_ori_con.groupby('ori')['act'].mean()).reset_index()
                    mean_act = mean_act_by_ori.act
                    ori = ori*(np.pi/180.0)

                    # creating a list of 10 initial starting point
                    # 9 random ones and one set (a, b, b2, c, theta_p) intelligent guess
                    # here a, b, b2, c, theta_p correspond to the initial guesses of the model params 
                    ini_rang = 9 
                    a = min(mean_act)
                    b = (max(mean_act)-min(mean_act))/2.0
                    b2 = max(mean_act)/2.0
                    c = 90*(np.pi/180.0)
                    theta_p = max(mean_act)*(np.pi/180.0)

                    # initial guess for model with 5 params
                    x0_5p = [0.01 + 0.2 * np.random.randint(1, 5, 5) for x in range(ini_rang)]
                    x5p = [np.array([a, b, c, theta_p, b2])]
                    # initial guess for model with 4 params
                    x0_4p = [0.01 + 0.2 * np.random.randint(1, 4, 4) for x in range(ini_rang)]
                    x4p = [np.array([a, b, c, theta_p])]

                    # generate the x0 array by combing the intelligent guess with the random one
                    x0_arr = list(itertools.chain(x0_5p, x5p)) if model == 'm5p' \
                        else list(itertools.chain(x0_4p, x4p))

                    # loss function (objective function too)
                    def L_fun(x):
                        if model == 'm5p':
                            J = np.mean((act - model_fun2(x, ori))**2.0)
                        else:
                            J = np.mean((act - model_fun1(x, ori))**2.0)
                        return J           

                    # get size of parameter vector from first element in x0_arr
                    params = np.empty(len(x0_arr[0]))
                    min_obj = np.Inf
                    # fit tuning curves, using random restart
                    for x0 in x0_arr:
                        res = minimize(L_fun, x0, options={'disp':False})
                        if res.fun < min_obj:
                            params = res.x
                            min_obj = res.fun

                    key['neuro_id'] = n_id
                    key['viscon'] = con_id['viscon']
                    key['model_id'] = model                 
                    key['params'] = params
                    key['act_mean_per_ori'] = list(mean_act)
                    self.insert1(key)

            

#--------------------------------------------------------Models-------------------------------------------------------

# model function 1
def model_fun1(x, theta):
    """
    Model: von Mises with two fixed peak heights and take 4 parameters. 
    inputs: 
    x: an array   # list of params (values)
    theta: float  # the orientation

    Return: predicted mean activity of the given orientation
    """
    return x[0] + (x[1] * np.exp(x[2] * np.cos(2.0 * (theta - x[3]))))

# model function 2    
def model_fun2(x, theta):
    """
    Model: von Mises with two unfixed peak heights and take 5 parameters. 
    inputs: 
    x: an array  # list of params (values)
    theta: float # the orientation

    Return: predicted mean activity of the given orientation
    """    
    return x[0] + (x[1] * np.exp(x[2] * np.cos(theta - x[3]))) + (x[4] * \
        np.exp(-x[2] * np.cos(theta - x[3])))

#--------------------------------------------------------Helper Methods------------------------------------------------

# auto populate: call this function to populate tables with new data
def populate_new_data():
    """
    This method is used for populating the 'Stimulus', \
    'ActivityStatistics' and 'TuningCurveFits' tables. 
    It calls the make methods implemented in the respective
    classes.
    """
    Stimulus.populate(display_progress=True)
    ActivityStatistics.populate(display_progress=True)
    TuningCurveFits.populate(display_progress=True)
    
    print("All tables are populated.")


# helps pick a model based on input
# (currently) it is called when ploting figures 
def pick_model(x, phi, model):
    """
    This method is used to pick a model type based on the input \
    When called it will run the model that was pick

    inputs:
    x: an array  # list of params (values)
    phi: float   # orientation in radians
    model: str   # a string with values 'm4p' or 'm5p' to indicate \
                    which model you want to plot
    return:
    predicted mean activity per orientation
    """
    if model == 'm5p':
        model = model_fun2(x, phi)
    else:
        model = model_fun1(x, phi)
    return model

# this method is called within the method
# 'tuningCurvePerModel' Since not all sessions have two contrast
# there are two possible values for contrast: low=0, high=1
# we need to check if the entered contrast is in the session
def check_contrast(mouse_id, session_date, viscon):
    """"
    The method checks if the contrast value entered is \
    in the contrasts list of the given session.

    inputs:
    mouse_id: int     # unique to each mouse
    session_date: date  # session date given by the date of the session
    viscon: int     # Trial value indicating the contrast of visual stimuli

    return:
    contrast
    """

   # get a list viscon values from session
    contrasts = [con['viscon'] for con in \
        (dj.U('viscon') 
        & (Stimulus 
        & f'mouse_id={mouse_id}' 
        & f'session_date="{session_date}"'))]

    # chech if the contrast is in the list (contrasts)
    # if not present, prompt user to enter the available
    # contrast value else just return the entered contrast value
    while True:
        if viscon not in contrasts:
            print("The visual contrast you entered is not in this session")
            print(f'The available contrast(s) in AJ0{mouse_id}_{session_date} is: {contrasts}')
            answer = int(input('To continue, enter the available contrast options:'))
            if answer in contrasts:
                viscon = answer
                print(f'your new visual contrast is: {viscon}')
                break
        else:
            break
    return viscon


#-------------------------------------plots---------------------------------------------------------

# plot of tuning curves
# this function plots both high and low contrasts on the same figure
def tuningCurvePerCon(mouse_id, session_date, model):
    """
    Plots subplots of random neurons comparing different \
    contrast values. The left and right y-axes have different scales

    inputs:
    mouse_id: int     # unique to each mouse
    session_date: date  # session date given by the date of the session
    model: varchar(5) # Type of model being fitted 

    returns:
    subplots of tuning curves
    """


    # get the number of neurons per_session
    # and randonly sample neuroId from this range
    num_of_plots = 9
    n_neurons_sess = len(dj.U('neuro_id') & 
        (Neuralactivity & f'mouse_id={mouse_id}' 
            & f'session_date="{session_date}"'))
    rand_nid = np.random.randint(0, n_neurons_sess, num_of_plots)

    # get the orientations to use as x-axis ticks
    ori_mean = [
        o['visori']*(np.pi/180.0)
        for o in (dj.U('visori') & ActivityStatistics)
        ]

    fig, axes = plt.subplots(3, 3, figsize=(15,10))
    axes = axes.ravel()  # array to 1D
    # loop through the randomly sampled neurons and axes
    for n_id, ax1 in zip(rand_nid, axes):
        # fetch the vison, parameters and mean activites from BD
        viscon, params, mean_act = (TuningCurveFits \
                    & f'mouse_id={mouse_id}' \
                    & f'session_date="{session_date}"' \
                    & f'neuro_id={n_id}' \
                    & f'model_id="{model}"'\
                ).fetch('viscon', 'params', 'act_mean_per_ori')

        phi = np.linspace(0,2*np.pi, 100)

        # plot low (or whatever is present if len(viscon) == 1) viscon plots
        color = 'tab:grey'
        lowConMean = ax1.plot(ori_mean, mean_act[0], \
            label='mean_act_con_0', 
            color=color, 
            ls='--')
        y0 = pick_model(params[0], phi, f'{model}')
        color = 'tab:red'
        ax1.set_title(f'Neuron {n_id}')
        ax1.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
        ax1.set_xticklabels([45, 90, 135, 180, 225, 270, 315, 360])
        ax1.set_xlabel('Orientation($\\Theta$)')
        ax1.set_ylabel('$\\Delta$F/F', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        lowConPlot = ax1.plot(phi, y0, label=f'viscon={viscon[0]}', color=color)

        # plot high viscon 
        if len(viscon) > 1:
            # high vicon plots
            ax2 = ax1.twinx()   # instantiate a second axes that shares the same x-axis
            color = 'tab:grey'
            highConMean = ax2.plot(ori_mean, mean_act[1], 
                label='mean_act_con_1', 
                color=color,
                ls='-'
            )
            y1 = pick_model(params[1], phi, f'{model}')
            color = 'tab:blue'
            ax2.tick_params(axis='y', labelcolor=color)
            highConPlot = ax2.plot(phi, y1, label=f'viscon={viscon[1]}', color=color)
            # labels for the legend for both contrasts
            leg = lowConMean + lowConPlot + highConMean + highConPlot 
        else:
            leg = lowConMean + lowConPlot  # labels for the legend for single contrast

    labs = [l.get_label() for l in leg] # get labels

    fig.suptitle(
        f"{num_of_plots} Randomly Fitted Tuning Curves for Model={model} in AJ0{mouse_id}_{session_date}",
        y=1.05, fontsize='xx-large'
    )
    
    fig.legend(leg, labs, loc='upper left')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

# plot of tuning curves
def tuningCurvePerModel(mouse_id, session_date, viscon):
    """
    Plots subplots of random neurons comparing different \
    contrast values. The left and right y-axes have different scales

    inputs:
    mouse_id: int     # unique to each mouse
    session_date: date  # session date given by the date of the session
    viscon: int     # trial value indicating the contrast of visual stimuli

    returns:
    subplots of tuning curves
    """

    # Since not all sessions have two contrast
    # we need to check if the entered contrast is in the session
    viscon = check_contrast(mouse_id, session_date, viscon)

    # get the number of neurons per_session
    # and randonly sample neuroId from this range
    num_of_plots = 16
    n_neurons_sess = len(dj.U('neuro_id') & 
        (Neuralactivity & f'mouse_id={mouse_id}' 
            & f'session_date="{session_date}"'))
    rand_nid = np.random.randint(0, n_neurons_sess, num_of_plots)

    # get the orientations to use as x-axis ticks
    ori_mean = [
        o['visori']*(np.pi/180.0)
        for o in (dj.U('visori') & ActivityStatistics)
        ]

    fig, axes = plt.subplots(4, 4, figsize=(15,10))
    axes = axes.ravel()  # array to 1D
    # loop through the randomly sampled neurons and axes
    for n_id, ax in zip(rand_nid, axes):
        # fetch the model, parameters and mean activites from BD
        model, params, mean_act = (TuningCurveFits \
                    & f'mouse_id={mouse_id}' \
                    & f'session_date="{session_date}"' \
                    & f'neuro_id={n_id}' \
                    & f'viscon={viscon}'\
                ).fetch('model_id', 'params', 'act_mean_per_ori')

        phi = np.linspace(0,2*np.pi, 100)
        ax.plot(ori_mean, mean_act[0], label='mean_activity')

        # for each model and parameter set plot the tuning curve
        for model, params in zip(model, params):
            y = pick_model(params, phi, f'{model}')
         
            ax.plot(phi, y, label=f'{model}')
            ax.set_title(f'Neuron {n_id}')
            ax.label_outer()
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right') 
    # add a big axis, hide frame
    axfig = fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(
        labelcolor='none', 
        which='both', 
        top=False, 
        bottom=False, 
        left=False, 
        right=False
        )
    fig.suptitle(
        f"{num_of_plots} Randomly Fitted Tuning Curves \
            for Viscon={viscon} in AJ0{mouse_id}_{session_date}",
        y=0.94, fontsize='xx-large'
    )
    plt.xlabel('Orientation', 
        fontdict={
            'fontsize': 20, 'fontweight': 'medium'
            }
        )
    plt.ylabel('$\\Delta$F/F', 
        fontdict={
            'fontsize': 20, 'fontweight': 'medium'
            }
        )
    # Tweak spacing to prevent clipping of ylabel
    axfig.yaxis.set_label_coords(-0.05,0.5)
