import os
import argparse

from utils.functions import Storage

class ConfigClassification():
    def __init__(self, args):
        # hyper parameters for models
        HYPER_MODEL_MAP = {
            'cmcm': self.__CMCM
        }
        # hyper parameters for datasets
        self.root_dataset_dir = args.root_dataset_dir
        HYPER_DATASET_MAP = self.__datasetCommonParams()

        # normalize
        model_name = str.lower(args.modelName)
        dataset_name = str.lower(args.datasetName)
        # load params
        commonArgs = HYPER_MODEL_MAP[model_name]()['commonParas']
        dataArgs = HYPER_DATASET_MAP[dataset_name]
        dataArgs = dataArgs['aligned'] if (commonArgs['need_data_aligned'] and 'aligned' in dataArgs) else dataArgs['unaligned']
        # integrate all parameters
        self.args = Storage(dict(vars(args),
                            **dataArgs,
                            **commonArgs,
                            **HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name],
                            ))
    
    def __datasetCommonParams(self):
        root_dataset_dir = self.root_dataset_dir
        tmp = {
            'iemocap':{
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'IEMOCAP'),
                    'seq_lens': (84, 157, 32),
                    # (text, audio, video)
                    'feature_dims': (4096, 64, 64),
                    'train_samples': 5240,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'weight_F1'
                }
            },
            'meld':{
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MELD'),
                    'seq_lens': (65, 157, 32),
                    # (text, audio, video)
                    'feature_dims': (4096, 64, 64),
                    'train_samples': 9992,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'weight_F1'
                }
            },
            'cherma':{
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'CHERMA0723'),
                    # (batch_size, seq_lens, feature_dim)
                    'seq_lens': (78, 543, 16), # (text, audio, video)
                    'feature_dims': (4096, 1024, 2048), # (text, audio, video)
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'cn',
                    'KeyEval': 'weight_F1',
                }
            },


        }
        return tmp

    def __CMCM(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_label_prefix':True,
                'need_normalized': False,
                'use_PLM': True,
                'save_labels': False,
            },
            # dataset
            'datasetParas':{
                'meld':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'task_specific_prompt': 'Please recognize the emotion of the above multimodal content from the target \
                                                set <neutral:0, surprise:1, fear:2, sadness:3, joy:4, disgust:5, anger:6>. response: The emotion is',
                    'max_new_tokens': 2,
                    'pseudo_tokens': 4,
                    'label_index_mapping': {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5,
                                           'anger': 6},
                    'batch_size': 8,
                    'learning_rate': 5e-5,
                    # feature subNets
                    'a_lstm_hidden_size': 64,
                    'v_lstm_hidden_size': 32,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    'warm_up_epochs': 90,
                    #loss weight   best：1
                    'gamma':1,
                    'update_epochs': 1,
                    'early_stop': 8,
                    # res
                    'H': 3.0
                },
                'cherma':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'task_specific_prompt': '请选择适用于上述多模态内容的情绪标签：<愤怒:0, 厌恶:1, 恐惧:2, 高兴:3, 平静:4, 悲伤:5, 惊奇:6>。响应: 情绪为',
                    'max_new_tokens': 2,
                    'pseudo_tokens': 4,
                    'label_index_mapping': {'愤怒': 0, '厌恶': 1, '恐惧': 2, '高兴': 3, '平静': 4, '悲伤': 5,
                                            '惊奇': 6},
                    'batch_size': 8,
                    'learning_rate': 5e-5,
                    # feature subNets
                    'a_lstm_hidden_size': 32,
                    'v_lstm_hidden_size': 16,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    'warm_up_epochs': 30,
                    'update_epochs': 1,
                    'early_stop': 8,
                    # loss weight
                    'gamma': 0,
                    # res
                    'H': 1.0
                },
            },
        }
        return tmp

    def get_config(self):
        return self.args