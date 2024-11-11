# Neural Wave Hackaton submission
## Team Hackabros - AI4Privacy

You can find project related trained models and weights in the below link 

[Piidgeon Agent](https://huggingface.co/hyacinthum/Piidgeon)

To use the model, use the following to load it directly from huggingface:
```python
from safetensors.torch import load_file
miniagent_model = AutoModelForTokenClassification.from_pretrained('hyacinthum/Piidgeon')

miniagent_tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-multilingual-cased")
miniagent_model.config.id2label = {0: 'O', 1: 'I-ACCOUNTNUM', 2: 'I-IDCARDNUM'}
miniagent_model.config.label2id = {'O': 0, 'I-ACCOUNTNUM': 1, 'I-IDCARDNUM': 2}
```

Or alternatively, from the GitHub repository:
```python
from safetensors.torch import load_file
config = AutoConfig.from_pretrained('pii_model/checkpoints/best_model')
miniagent_model = AutoModelForTokenClassification.from_config(config)
state_dict = load_file('pii_model/checkpoints/best_model/model.safetensors')
miniagent_model.load_state_dict(state_dict)

miniagent_tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-multilingual-cased")
miniagent_model.config.id2label = {0: 'O', 1: 'I-ACCOUNTNUM', 2: 'I-IDCARDNUM'}
miniagent_model.config.label2id = {'O': 0, 'I-ACCOUNTNUM': 1, 'I-IDCARDNUM': 2}
```

To run the evalutation of model please use below code :
```bash
python inference.py -s path/to/evalutaion_file/example.jsonl -m pii_model
```

## Video showcase

[Video link](https://youtu.be/leFG8oPcXs0)

