import os
import argparse

from utils.functions import Storage

class ConfigRegression():
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
            'mosi':{
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/unaligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (4096, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'MAE'
                }
            },
            'mosei':{
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/unaligned_50.pkl'),
                    'seq_lens': (50, 500, 375),
                    # (text, audio, video)
                    'feature_dims': (4096, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'MAE'
                }
            },


            'simsv2': {
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'SIMS_V2/ch-simsv2s.pkl'),
                    # (batch_size, seq_lens, feature_dim)
                    'seq_lens': (50, 925, 232),  # (text, audio, video)
                    'feature_dims': (4096, 25, 177),  # (text, audio, video)
                    'train_samples': 2722,
                    'num_classes': 3,
                    'language': 'cn',
                    'KeyEval': 'MAE',
                }
            }
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
                'mosei':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'task_specific_prompt': 'Please predict the sentiment intensity of the above multimodal content in the range [-3.0, 3.0]. response: The sentiment is',
                    'max_new_tokens': 4,
                    'pseudo_tokens': 4,
                    'batch_size': 8,
                    'learning_rate': 5e-5,
                    # feature subNets
                    'a_lstm_hidden_size': 64,
                    'v_lstm_hidden_size': 32,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    'warm_up_epochs':30,
                    #loss weight   best：1
                    'gamma':1,
                    'update_epochs': 1,
                    'early_stop': 10,     #10和8没啥区别
                    # res
                    'H': 3.0
                },

                'simsv2': {
                    # the batch_size of each epoch is update_epochs * batch_size
                    'max_new_tokens': 4,
                    'pseudo_tokens': 4,
                    'task_specific_prompt': '请对上述多模态内容的情感强度进行预测，范围在[-1.0, 1.0]之间。响应: 情感为',
                    'batch_size': 8,
                    'learning_rate': 5e-5,
                    # feature subNets
                    'a_lstm_hidden_size': 64,
                    'v_lstm_hidden_size': 64,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    'warm_up_epochs': 80,
                    'update_epochs': 1,
                    'early_stop': 10,
                    # loss weight  best：0.25
                    'gamma': 1,
                    # res
                    'H': 1.0
                },
            },
        }
        return tmp

    def get_config(self):
        return self.args