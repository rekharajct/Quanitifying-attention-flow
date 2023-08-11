import torch

def generate_bert_input(dataset_obj, sentence, target=None):
    """
    Returns the BERT input tensors
    param dataset_obj: object of train dataset
    param sentence: masked sentence
    param target: SV number target. default None
    """
    #get vocabulary used for BERT training
    vocab  = dataset_obj.vocab

    #sent to list
    sent_list = sentence.split()

    #tokenize
    tokens = [vocab[w] for w in sent_list]
    
    
    #print(sv_out)
    # add CLS and SEP
    tokens = [dataset_obj.CLS] + tokens + [dataset_obj.SEP]

    #pad to seq_len
    padding = [dataset_obj.PAD for _ in range(dataset_obj.seq_len - len(tokens))]
    tokens.extend(padding)

    if target:
        # Define the mapping of targets and convert to numbers
        mapping = {'VBP': 1, 'VBZ': 0} 
        sv_out = [mapping[target]]

        return {'input': torch.Tensor(tokens).long(),
                'sv_out': torch.Tensor(sv_out)}
    else:
        return {'input': torch.Tensor(tokens).long()}
    

def mlm_predict(dataset_obj, bert_model, sentence, mask_index, target=None):
    """
    Predict the masked word using BERT
    param sentence: masked sentence
    param index: index of masked token 
    param target: SV number prediction target
    """
    bert_inp = generate_bert_input(dataset_obj, sentence, target)
    print(bert_inp["input"])

    #get bert_out: dict{}
    bert_out = bert_model(bert_inp["input"])
    bert_inp_grad = bert_out["inp_grad"]
    print(bert_inp_grad)
    mlm_out = bert_out["bert_out"]
    attn = bert_out["attention"]
    hidden_states = bert_out["hidden_states"]
    
    mlm_out = torch.squeeze(mlm_out)
    soft_max_layer = torch.nn.Softmax(dim=1)
    tokens_logits = soft_max_layer(mlm_out)
    tokens_pred = torch.argmax(tokens_logits, dim=1)
    tokens_pred_list = list(tokens_pred.detach().numpy())
    tokens_pred_words = [dataset_obj.rvocab[t] for t in tokens_pred_list]

    return tokens_pred_words[mask_index+1]
