import re
import os
from tensorboard.backend.event_processing import event_accumulator


class read_tblogs():
    # read scalar values from tensorboard logs

    def __init__(self,filepath,match=''):
        self.filepath = filepath # tensorboard logdir
        self.match = match  # reg-expression for files in logdir
    
    def val_scalars(self):
        #returns all scalar values corresponding to the validation set

        values_dict = dict()
        for model_name in os.listdir(self.filepath):
            if re.match(self.match,model_name) != None:
                model_f = os.path.join(self.filepath,model_name)
                try:
                    ac = event_accumulator.EventAccumulator(path= os.path.join(model_f,'validation'))
                    ac.Reload()           
                    scalar_list = ac.Tags()['scalars']
                    
                    temp_dict = dict()
                    for scalar in scalar_list:
                        temp_dict[scalar] = list()            
                        for ac_loss in ac.Scalars(scalar):
                                temp_dict[scalar].append(ac_loss.value)
                    values_dict[model_name] = temp_dict
                except Exception as p:
                    print(p)
        return values_dict 
                
    def train_scalars(self):
        #returns all scalar values corresponding to the training set
        values_dict = dict()
        for model_name in os.listdir(self.filepath):
            if re.match(self.match,model_name) != None:
                model_f = os.path.join(self.filepath,model_name)
                try:
                    ac = event_accumulator.EventAccumulator(path= os.path.join(model_f,'train'))
                    ac.Reload()           
                    scalar_list = ac.Tags()['scalars']
                    
                    temp_dict = dict()
                    for scalar in scalar_list:
                        temp_dict[scalar] = list()            
                        for ac_loss in ac.Scalars(scalar):
                                temp_dict[scalar].append(ac_loss.value)
                    values_dict[model_name] = temp_dict
                except Exception as p:
                    print(p)
        return values_dict
