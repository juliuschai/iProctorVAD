import torch
torch.set_num_threads(1)
from pprint import pprint

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)

(get_speech_ts,
 get_speech_ts_adaptive,
 _, read_audio,
 _, _, _) = utils

files_dir = torch.hub.get_dir() + '/snakers4_silero-vad_master/files'

wav = read_audio(f'{files_dir}/en.wav')
# full audio
# get speech timestamps from full audio file

# classic way
speech_timestamps = get_speech_ts(wav, model,
                                  num_steps=4)
pprint(speech_timestamps)

# adaptive way
speech_timestamps = get_speech_ts_adaptive(wav, model)
pprint(speech_timestamps)
