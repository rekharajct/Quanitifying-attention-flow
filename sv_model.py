import torch.nn as nn
import torch
class SVClassifier(nn.Module):
    """
    Classifier for SV prediction
    """
    def __init__(self,bert_model, num_classes, find_grad=None):
        """
        param bert_model: pretrained BERT
        param num_classes: num of classes 
        """
        super(SVClassifier,self).__init__()
        self.bert = bert_model
        self.linear = nn.Linear(bert_model.embed_size,num_classes)
        self.embeddings = nn.Embedding(bert_model.n_embeddings,bert_model. embed_size)
        self.softmax = nn.Softmax(dim = -1)
        self.find_grad = find_grad
        
    def forward(self,input_ids):
        """
        param input_ids: tokenized inputs
        return a dict of classifier logits, bert_embedding, attn_outs, hidden_states of BERT
        """

        # get a  BERT embedding, attention, hidden states
        # bert returns {"inp_grad":self.grad_dict, "bert_out":out, "attention":layerwise_attn, "hidden_states":layerwise_hidden_states}
        if self.bert.input_embedding:
            input_embedding = self.embeddings(input_ids)

            bert_output = self.bert(input_embedding) 
        else:
            bert_output = self.bert(input_ids)


        input_grad = bert_output["inp_grad"] #dict of input gradients 
        
        bert_embedding = bert_output["bert_embedding"] #bert embedding [batch_size x seq_len x embed_size]

        bert_embed_grads = bert_output["bert_emb_grads"] # gradient of bert embedding w.r.t softmax probs of vocab
    

        attn_outs = bert_output["attention"] #layerwise attention as list

        bert_predict_logits = bert_output["bert_prediction"]
    
        bert_hidden_states = bert_output["hidden_states"] # list of  bert encoder layer outputs [batch_size X seq_len X embedd_size]
    

        bert_cls = bert_hidden_states[-1][:,0] #get the embedding of CLS token

        
        logits = self.linear(bert_cls) # logits of classifier
        
        classifier_grad = {}
        
        #compute gradients of the logits w.r.t input _embedding
        if self.find_grad:
            for cl_ in range(logits.size(1)):
                classifier_grad[cl_] = torch.autograd.grad(outputs=logits[:, cl_], inputs=input_embedding, \
                                            grad_outputs =torch.ones_like(logits[:, cl_]), retain_graph=True)[0]
            
            
        probs = self.softmax(logits) #softmax probabilies of classifier predictions

        return {"logits":logits, "softmax_probs": probs,  "bert_embedding":bert_embedding,\
                "attentions":attn_outs, "bert_hidden_states":bert_hidden_states, "bert_predict_logits": bert_predict_logits,\
                "input_grad": input_grad, "bert_embed_grads": bert_embed_grads, "classifier_grad": classifier_grad}
    

    
