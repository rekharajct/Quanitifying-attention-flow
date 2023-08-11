import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from model import *
from optim_schedule import *
from sv_model import *


import tqdm

class SVTrainer:
    """
    SVTrainer use the pretrained BERT model to train a Subject Verb number prediction classifier.

       
    """

    def __init__(self, bert: BERT, num_classes: 2,
                 train_dataloader: DataLoader, 
                 test_dataloader: DataLoader = None,
                 svc_model_state_dict=None,
                 find_grad = None,
                 lr: float = 1e-4, betas=(0.9, 0.999),
                 weight_decay: float = 0.01, 
                 warmup_steps=10000,
                 with_cuda: bool = True, 
                 cuda_devices=None,                 
                 log_freq: int = 10
                 ):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param num_classes: number of classes in classification
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device('cuda' if cuda_condition else 'cpu')

        # This BERT model will be saved every epoch
        self.bert_model = bert

        # Classifier model with bert and num_classes as input
        self.model = SVClassifier(self.bert_model,num_classes, find_grad)

        if svc_model_state_dict: # if train from a pretrained model
            self.model.load_state_dict(svc_model_state_dict)
        
        
        self.with_cuda = with_cuda

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)
        
        self.model = self.model.to(self.device)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optimizer = AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        
        #warmup
        #self.optim_schedule = ScheduledOptim(self.optimizer,   self.bert_model.embed_size, n_warmup_steps=warmup_steps)
        self.optim_schedule = ScheduledOptim(self.optimizer,   num_classes, n_warmup_steps=warmup_steps)

        # Using cross entropy loss  function for predicting the masked_token.
        # Cross entropy combined softamax and NLL. so should not apply softmax
        self.criterion = nn.CrossEntropyLoss() 
        #self.criterion = nn.NLLLoss(ignore_index=0)

        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        
        

    def train(self, epoch):
        self.model.train()
        self.iteration(epoch, self.train_data)
        return self.model

    def test(self, epoch):
        self.model.eval()
        self.iteration(epoch, self.test_data, train=False)    



    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")
        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        for i, data in data_iter:
            #get a batch of data
            batch = {key: value.to(self.device) for key, value in data.items()}

            #get masked inputs
            
            input = batch['input'] 
    
            #get targets for masked inputs
            target = batch['sv_target'] 
        

            #get the logits for inputs : shape[batch_size, num_class]
            #the classifier outputs a dict {"logits", "softmax_probs","bert_embedding","attentions",
            # "bert_hidden_states","input_grad"}
            classifier_output = self.model.forward(input)
            logits = classifier_output["logits"]
        
            
            probs = classifier_output["softmax_probs"]
            bert_embedding = classifier_output["bert_embedding"]
            layerwise_attentions = classifier_output["attentions"]
            bert_hidden_states = classifier_output["bert_hidden_states"]
            input_grad = classifier_output["input_grad"]           
            
            

        
            #compute classification  loss
            target_flattened = torch.flatten(target) #flatten the targets
            classfn_loss = self.criterion(logits, target_flattened )

            #backpropagation
            if train:
                self.optim_schedule.zero_grad()
                classfn_loss.backward()
                self.optim_schedule.step_and_update_lr()
            
            
                
                
             
            
            
            
            #update average loss
            avg_loss += classfn_loss.item()

            #--------------update classification accuracy-------
            #get the classes predicted
            pred_classes  =  torch.argmax(probs, dim=1)
            
            #compute average accurcy
            total_correct += torch.sum(pred_classes.data==target_flattened).float()
            total_element += target.nelement() 
            avg_acc =  (total_correct / total_element * 100.0 ).item()         


            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "avg_acc": avg_acc,
                "loss": classfn_loss.item()
            }

       
            print("data_iter_len", len(data_iter))
            print(post_fix)

        
        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter), "avg_acc=",avg_acc)


    def save(self, epoch, file_path="saved_models/bert_trained"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        #output_path = file_path
        output_path = file_path + ".ep%d" % epoch
        if self.with_cuda and torch.cuda.device_count() > 1:
        	torch.save(self.model.module.state_dict(), output_path)
        else:
        	torch.save(self.model.state_dict(), output_path)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path


    




