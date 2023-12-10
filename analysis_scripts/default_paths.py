import os

ROOT_REPO_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
WORD_AND_SPEAKER_ENCODINGS_PATH = os.path.join(ROOT_REPO_DIR, 'robustness', 'audio_functions', 'word_and_speaker_encodings_jsinv3.pckl')
ASSETS_PATH = os.path.join(ROOT_REPO_DIR, 'assets')
WORDNET_ID_TO_HUMAN_PATH = os.path.join(ROOT_REPO_DIR, 'analysis_scripts', 'wordnetID_to_human_identifier.txt')
IMAGENET_PATH = '/om/data/public/imagenet/images_complete/ilsvrc/'
if not os.path.exists(IMAGENET_PATH):
    IMAGENET_PATH = None
    print('### WARNING: UNABLE TO FIND IMAGENET FILES. IF TRANING IMAGENET MODELS, CHANGE PATH SPECIFIED IN analysis_scripts/default_paths.py. METAMERS CAN BE GENERATED WITHOUT THESE FILES. ###')
JSIN_PATH = '/om4/group/mcdermott/projects/ibmHearingAid/assets/data/datasets/JSIN_v3.00/nStim_20000/2000ms/rms_0.1/noiseSNR_-10_10/stimSR_20000/reverb_none/noise_all/JSIN_all_v3/subsets'
if not os.path.exists(JSIN_PATH):
    JSIN_PATH = None
    print('### WARNING: UNABLE TO FIND JSIN AUDIO TRAINING DATASET FILES. IF TRAINING AUDIO MODELS, CHANGE PATH SPECIFIED IN analysis_scripts/default_paths.py. METAMERS CAN BE GENERATED WITHOUT THESE FILES ###')

# fMRI dataset paths
fMRI_DATA_PATH = os.path.join(ASSETS_PATH, 'fMRI_natsound_data')
