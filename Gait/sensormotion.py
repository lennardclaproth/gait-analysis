import os
import matplotlib.pyplot as plt
import numpy as np
import sensormotion as sm
import pandas as pd

cwd = os.getcwd()
path = cwd + '/sensormotion/'
files = []
freq = 9  # Filter frequency in Hz
fo = 4  # Filter order

# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.csv' in file:
            files.append(os.path.join(r, file))

for f in files:
    force = pd.read_csv(f, sep=';')
    force.columns = ['index', 'gFx', 'gFy', 'gFz']
    force.index.names = ['time_index']
    force = force.drop("index", axis=1)

    gFx = force.loc[:, 'gFx']
    gFx = gFx.reset_index()
    gFx = gFx.drop('time_index', 1)
    x = gFx.values.squeeze()

    gFy = force.loc[:, 'gFy']
    gFy = gFy.reset_index()
    gFy = gFy.drop('time_index', 1)
    y = gFy.values.squeeze()

    gFz = force.loc[:, 'gFz']
    gFz = gFz.reset_index()
    gFz = gFz.drop('time_index', 1)
    z = gFz.values.squeeze()

    # Simulated time values
    sampling_rate = 30  # samples per second
    time_per_sample = 1/sampling_rate*1000
    np.random.seed(123)
    # Time values on x-axis on plots
    time = np.arange(0, x.__len__()) * time_per_sample  # time in ms

    # Plot
    sm.plot.plot_signal(time, [{'data': x, 'label': 'Medio-lateral (ML) - side to side', 'line_width': 0.5},
                               {'data': y, 'label': 'Vertical (VT) - up down', 'line_width': 0.5},
                               {'data': z, 'label': 'Antero-posterior (AP) - forwards backwards', 'line_width': 0.5}],
                        subplots=True, fig_size=(10, 7))

    # filter dominant frequencies
    _ = sm.signal.fft(y, sampling_rate, plot=True)

    # plot filter response curve
    sm.plot.plot_filter_response(freq, sampling_rate, 'low', filter_order=fo)

    # Build the filter
    b, a = sm.signal.build_filter(freq, sampling_rate, 'low', filter_order=fo)

    # Filter signals
    x_f = sm.signal.filter_signal(b, a, x)  # ML medio-lateral
    y_f = sm.signal.filter_signal(b, a, y)  # VT vertical
    z_f = sm.signal.filter_signal(b, a, z)  # AP antero-posterior


    # Create plots with overlaid filtered signals (in red)
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 9))

    ax[0].set_title('Medio-lateral (ML) - side to side')
    ax[0].plot(time, x, linewidth=0.3, color='k')
    ax[0].plot(time, x_f, linewidth=0.8, color='r')

    ax[1].set_title('Vertical (VT) - up down')
    ax[1].plot(time, y, linewidth=0.3, color='k')
    ax[1].plot(time, y_f, linewidth=0.9, color='r')

    ax[2].set_title('Antero-posterior (AP) - forwards backwards')
    ax[2].plot(time, z, linewidth=0.3, color='k')
    ax[2].plot(time, z_f, linewidth=0.9, color='r')

    fig.subplots_adjust(hspace=.5)


    # cadence and step time extraction
    peak_times, peak_values = sm.peak.find_peaks(time, y_f, peak_type='valley', min_val=0.6, min_dist=10, plot=True)

    step_count = sm.gait.step_count(peak_times)
    cadence = sm.gait.cadence(time, peak_times)
    step_time, step_time_sd, step_time_cov = sm.gait.step_time(peak_times)

    print('------------------------------------------')
    print(f)
    print('------------------------------------------')
    print(' - Number of steps: {}'.format(step_count))
    print(' - Cadence: {:.2f} steps/min'.format(cadence))
    print(' - Mean step time: {:.2f}ms'.format(step_time))
    print(' - Step time variability (standard deviation): {:.2f}'.format(step_time_sd))
    print(' - Step time variability (coefficient of variation): {:.2f}'.format(step_time_cov))


    # stride regularity and step symmetry extraction
    ac, ac_lags = sm.signal.xcorr(y_f, y_f, scale='unbiased', plot=True)

    ac_peak_times, ac_peak_values = sm.peak.find_peaks(ac_lags, ac, peak_type='peak', min_val=0.1, min_dist=30, plot=True)

    step_reg, stride_reg = sm.gait.step_regularity(ac_peak_values)
    step_sym = sm.gait.step_symmetry(ac_peak_values)

    print(' - Step regularity: {:.4f}'.format(step_reg))
    print(' - Stride regularity: {:.4f}'.format(stride_reg))
    print(' - Step symmetry: {:.4f}'.format(step_sym))

print('------------------------------------------')



