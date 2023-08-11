import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from model import *
from optim_schedule import *


from tqdm import tqdm as tqdm 

class BERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with MLM training method.

       
    """

    def __init__(self, bert: BERT, vocab_size: int,
                 train_dataloader: DataLoader, 
                 test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999),
                 weight_decay: float = 0.01, 
                 warmup_steps=10000,
                 with_cuda: bool = True, 
                 cuda_devices=None, 
                 log_freq: int = 10):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device('cuda' if cuda_condition else 'cpu')

        # This BERT model will be saved every epoch
        self.bert_model = bert.to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.bert_model = nn.DataParallel(self.bert_model)
            
        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optimizer = AdamW(self.bert_model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        
        #warmup
        self.optim_schedule = ScheduledOptim(self.optimizer, \
                                             self.bert_model.module.embed_size, n_warmup_steps=warmup_steps)

        # Using cross entropy loss  function for predicting the masked_token.
        # Cross entropy combined softamax and NLL. so should not apply softmax
        self.criterion = nn.CrossEntropyLoss(ignore_index=0) 
        #self.criterion = nn.NLLLoss(ignore_index=0)

        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.bert_model.parameters()]))

    def train(self, epoch):
        self.bert_model.train()
        self.iteration(epoch, self.train_data)
        return self.bert_model

    def test(self, epoch):
        self.bert_model.eval()
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
        data_iter = tqdm(enumerate(data_loader),desc="EP_%s:%d" % (str_code, epoch),total=len(data_loader),bar_format="{l_bar}{r_bar}" )
        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        for i, data in data_iter:
            #get a batch of data
            batch = {key: value.to(self.device) for key, value in data.items()}

            #get masked inputs
            mlm_input = batch['input'] 

            #get targets for masked inputs
            mlm_target = batch['target'] 

            #print(mlm_input.min(), mlm_input.max())
            
            #get the encoding using BERT will be of [seq_len X ]
            bert_output = self.bert_model.forward(mlm_input)
            
            bert_inp_grad = bert_output["inp_grad"]
            mlm_output = bert_output["bert_prediction"]
            attn_layerwise = bert_output["attention"]
            hidden_states_layerwise = bert_output["hidden_states"]
            


            #compute MLM loss
            mlm_loss = self.criterion(mlm_output.transpose(1, 2), mlm_target)

            #backpropagation
            if train:
                self.optim_schedule.zero_grad()
                mlm_loss.backward()
                self.optim_schedule.step_and_update_lr()
            
            #update average loss
            avg_loss += mlm_loss.item()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                #"avg_acc": total_correct / total_element * 100,
                "loss": mlm_loss.item()
            }

            print(post_fix)

        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter))


    def save(self, epoch, file_path="saved_models/bert_trained"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        #output_path = file_path
        output_path = file_path + ".ep%d" % epoch
        
        #torch.save(self.bert_model.cpu(), output_path)
        torch.save(self.bert_model.module.state_dict(), output_path)
        #self.bert_model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path


    





