'''
The EEG File here is written in plexon plx file, python neo might be able to open it.

'''

#%%
import OS_Tools_Kit as ot
import neo

wp = r'D:\_Shu_Data\20240220_#244_chat-flox\EEG'

eeg_file = ot.join(wp,'20240220_chat-flox_#244_001.plx')

#%%
from neo.io import PlexonIO
reader = PlexonIO(eeg_file)
blks = reader.read()

#%% Seperate eeg channel, get all series data and all trains.
