import os
from datetime import datetime
import torch

class ModelParams:
    ids_size = 320
    d_model = 512
    hidden_size = 256
    n_classes = 4+1+1
    d_ff = 2048
    N = 2
    n_heads = 16
    n_epochs = 200
    batch_size = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DataPath:
    input_data_name = 'src_wolf_320_pad.csv'
    label_data_name = 'tgt_wolf.csv'
    input_model_path = None
    path_of_here = os.path.dirname(os.path.abspath(__file__))

    place_of_data = '../data'
    data_path = os.path.join(path_of_here, place_of_data)
    input_data_path = os.path.join(data_path, input_data_name)
    label_data_path = os.path.join(data_path, label_data_name)
    
    place_of_archive = '../data'
    archive_path = os.path.join(path_of_here, place_of_archive)

    @classmethod
    def emit_losses_path(cls):
        return os.path.join(cls.archive_path, 'losses.csv')

    @classmethod
    def emit_output_model_path(cls):
        return os.path.join(cls.archive_path, 'classify.pt')
        
    @classmethod
    def emit_temp_model_path(cls, epochs):
        return os.path.join(cls.archive_path, 'classify_' + str(epochs) + '.pt')
