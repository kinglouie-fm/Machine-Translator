# BPE
## Learning
- To learn bpe, we must provide
- Then

## Applying BPE


# Dictionaries



# Model
## Training
NEEDED: 
- source and target files for training and development
- files that specify the bpe merge operations for source and target (each saved to vocabulary/merge_operations)
OPTIONAL:
- pre-trained model (path can be specified as load_model_path)

USE: 
- specify everything in a config file (e.g. training_config.json)
- execute
'''
python model/model.py model/training_config.json
'''

## Search
NEEDED:
- trained model (saved to model/saved_models)
- vocabulary for source and target (each saved to vocabulary/vocabulary)
- source file for translation (saved to test_data)

OPTIONAL:
- reference file (saved to test_data)

USE: 
- specify everything in a config file (e.g. search_config.json)
- execute
'''
python model/search.py model/search_config.json
'''

# BLEU
NEEDED:
- reference file (saved anywhere)
- hypothesis file (saved anywhere)

USE:
- execute
'''
python metrics/metrics.py test_data/search_config.json
'''