'''
The EEG File here is written in plexon plx file, python neo might be able to open it.

'''

#%%
import OS_Tools_Kit as ot
import neo
import numpy as np

wp = r'D:\#Shu_Data\LX\240305\Data_Pinch2WakeUp\Pinch_PlsStartHere\ChATFlox_PinchGood\20240220_#244_chat-flox\EEG'

eeg_file = ot.join(wp,'20240220_chat-flox_#244_001.plx')

#%% load plx file, this file's structure is as stupid as hell, you need 30G memory for 1Gb File and about 10min.
from neo.io import PlexonIO
reader = PlexonIO(eeg_file)
blks = reader.read()


#%% Seperate eeg channel, get all series data and all trains.
eeg_signals = blks[0].segments[0].analogsignals[0]
sampling_rate = eeg_signals.sampling_rate
raw_data = np.array(eeg_signals)
np.save(r'D:\#Shu_Data\LX\240305\Data_Pinch2WakeUp\Pinch_PlsStartHere\ChATFlox_PinchGood\20240220_#244_chat-flox\EEG\EEG_Python',raw_data)

