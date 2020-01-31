import yaml
import os
from util.librispeech_dataset import create_dataloader
#from util.lstm_functions import log_parser,batch_iterator, collapse_phn
from util.shortet_libri_lstm_function import log_parser,batch_iterator, collapse_phn
#from model.blstm_fc_mp import BLSTM
from model.dens_blstm import BLSTM
import numpy as np
from torch.autograd import Variable
import torch
import time
from tensorboardX import SummaryWriter
import math
import argparse


torch.cuda.set_device(0)

parser = argparse.ArgumentParser(description='Training script for LAS on Librispeech .')

parser.add_argument('config_path', metavar='config_path', type=str,
                     help='Path to config file for training.')

paras = parser.parse_args()

config_path = paras.config_path

# Load config file for experiment
print('Loading configure file at',config_path)
conf = yaml.load(open(config_path,'r'))

# Parameters loading
print()
print('Experiment :',conf['meta_variable']['experiment_name'])
total_steps = conf['training_parameter']['total_steps']

listener_model_path = conf['meta_variable']['checkpoint_dir']+conf['meta_variable']['experiment_name']+'.listener'
speller_model_path = conf['meta_variable']['checkpoint_dir']+conf['meta_variable']['experiment_name']+'.speller'
lstm_model_path = conf['meta_variable']['checkpoint_dir']+conf['meta_variable']['experiment_name']+'.lstm'
verbose_step = conf['training_parameter']['verbose_step']
valid_step = conf['training_parameter']['valid_step']
tf_rate_upperbound = conf['training_parameter']['tf_rate_upperbound']
tf_rate_lowerbound = conf['training_parameter']['tf_rate_lowerbound']
tf_decay_step = conf['training_parameter']['tf_decay_step']
seed = conf['training_parameter']['seed']

# Fix random seed
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Load preprocessed LibriSpeech Dataset
#/home/neda.ahmadi/DutchSpeechRecognition/clips_flac/new-train.csv
train_set = create_dataloader(conf['meta_variable']['data_path']+'/train-small.csv', 
                              **conf['model_parameter'], **conf['training_parameter'], shuffle=True,training=True)
valid_set = create_dataloader(conf['meta_variable']['data_path']+'/dev-small.csv',
                              **conf['model_parameter'], **conf['training_parameter'], shuffle=False,drop_last=True)

#train_set = create_dataloader(conf['meta_variable']['data_path']+'/train.csv',
#                              **conf['model_parameter'], **conf['training_parameter'], shuffle=True,training=True)
#valid_set = create_dataloader(conf['meta_variable']['data_path']+'/dev.csv',
#                              **conf['model_parameter'], **conf['training_parameter'], shuffle=False,drop_last=True)


idx2char = {}
with open(conf['meta_variable']['data_path']+'/idx2chap.csv','r') as f:
    for line in f:
        if 'idx' in line:continue
        idx2char[int(line.split(',')[0])] = line[:-1].split(',')[1]

# Load pre-trained model if needed
if conf['training_parameter']['use_pretrained']:
    #global_step = conf['training_parameter']['pretrained_step']
    global_step = 14095
    lstm = torch.load(conf['training_parameter']['pretrained_listener_path'])
    #listener = torch.load(conf['training_parameter']['pretrained_listener_path'])
    #speller = torch.load(conf['training_parameter']['pretrained_speller_path'])
else:
    global_step = 0
    lstm =BLSTM(**conf['model_parameter'])



optimizer = torch.optim.Adam([{'params':lstm.parameters()}],lr=conf['training_parameter']['learning_rate'])
#optimizer = torch.optim.SGD([{'params':lstm.parameters()}],lr=conf['training_parameter']['learning_rate'])
#optimizer = torch.optim.SGD([{'params':lstm.parameters()}],lr=0.008)


#print('Optimizer ADAM',optimizer)
#print('Optimizer SGD',lstm.parameters())

best_ler = 1.0
record_gt_text = False
log_writer = SummaryWriter(conf['meta_variable']['training_log_dir']+conf['meta_variable']['experiment_name'])

# Training
print('Training starts...',flush=True)


#### LIMITING RUNNING TIME BY PROCESSING FEWER BATCHES
# for DC sever
# bs4 * 14 = 1 min
# for Parag server
# bs8 * 10 = 1 min
#total_time_mins = 50.
#total_steps = 50000.
#batches_per_min = 100.
### under sampling
#time_per_iter = total_time_mins / total_steps
#max_count_batch = int(time_per_iter * batches_per_min)
max_count_batch = int(len(train_set) / conf['training_parameter']['batch_size'])
#max_count_batch = 1
###
print("max_count_batch: ", max_count_batch)
print("batch size: ", conf['training_parameter']['batch_size'])
print("len train_set: ", len(train_set))
#emp_mean = 8.5
#emp_std = 3.5
#emp_normalize = True
#########################################################

def pad(mat, to_length, dim):
    if mat.shape[dim] >= to_length:
        return mat
    pad = np.take(np.zeros(mat.shape), range(to_lenghth - mat.shape[dim]), axis=dim)
    return np.concatenate((pad, mat), axis=dim)

batch_step = 1
time_axis = 2
batch_axis = 1
character_cut  = 20
while global_step < total_steps:
    global_step += 1
    # Teacher forcing rate linearly decay
    tf_rate = tf_rate_upperbound - (tf_rate_upperbound-tf_rate_lowerbound)*min((float(global_step)/tf_decay_step),1)
    
    # Training
    train_loss = []
    train_ler = []
    batch_limit_counter = 0
    
    for batch_data, batch_label in train_set:
        #spectogram_cut = int(batch_data.shape[time_axis] * character_cut / batch_label.shape[time_axis])
        spectogram_cut = 140
        if batch_data.shape[time_axis] < spectogram_cut:
            continue     
        
        batch_lable_size = batch_label.size()[2]
        number_of_label_chunks = int(math.ceil(batch_lable_size/character_cut))
        number_of_spec_chunks = int(math.ceil(batch_data.shape[time_axis]/spectogram_cut))
        number_of_chunks = min(number_of_label_chunks, number_of_spec_chunks)
        #batch_label_chopped = np.concatenate([pad(batch_label[:,:,i*character_cut:(i+1)*character_cut,:], character_cut, time_axis) for i in range(number_of_chunks)], axis=2)
        batch_label_chopped = torch.cat([batch_label[:,:,i*character_cut:(i+1)*character_cut,:] for i in range(number_of_chunks-1)], batch_axis)
        batch_data_chopped = torch.cat([batch_data[:,:,i*spectogram_cut:(i+1)*spectogram_cut,:] for i in range(number_of_chunks-1)], batch_axis)
       
        print('Current step :', batch_step, end='\r',flush=True)
        #if emp_normalize:
        #    batch_data = (batch_data - emp_mean) / emp_std

        #batch_loss, batch_ler = batch_iterator(batch_data, batch_label, lstm, optimizer, tf_rate, is_training=True, data='libri', **conf['model_parameter'])
        batch_loss, batch_ler = batch_iterator(batch_data_chopped, batch_label_chopped, lstm, optimizer, tf_rate, is_training=True, data='libri', **conf['model_parameter'])
        train_loss.append(batch_loss)
        train_ler.extend(batch_ler)

        batch_step += 1
        batch_limit_counter += 1


        if global_step % valid_step == 0:
            break

        if batch_limit_counter >= max_count_batch:
            break
    
    # Validation
    val_loss = []
    val_ler = []
    batch_limit_counter = 0
    for batch_data,batch_label in valid_set:
       # if emp_normalize:
       #     batch_data = (batch_data - emp_mean) / emp_std

        batch_loss, batch_ler = batch_iterator(batch_data, batch_label, lstm, optimizer,tf_rate, is_training=False, data='libri', **conf['model_parameter'])
       
        val_loss.append(batch_loss)
        val_ler.extend(batch_ler)

        batch_limit_counter += 1
        if batch_limit_counter >= max_count_batch:
            break

    print('\n global step: ', global_step, flush=True)

    train_loss = np.array([sum(train_loss)/len(train_loss)])
    train_ler = np.array([sum(train_ler)/len(train_ler)])

    val_loss = np.array([sum(val_loss)/len(val_loss)])
    val_ler = np.array([sum(val_ler)/len(val_ler)])

    print('loss', {'train': train_loss}, 'cer', {'train': train_ler}, global_step, flush=True)
    print('loss', {'dev': val_loss}, 'cer', {'dev': val_ler}, global_step, flush=True)

    log_writer.add_scalars('loss', {'train': train_loss}, global_step)
    log_writer.add_scalars('cer', {'train': np.array([np.array(train_ler).mean()])}, global_step)

    log_writer.add_scalars('loss', {'dev': val_loss}, global_step)
    log_writer.add_scalars('cer', {'dev': val_ler}, global_step)


    ###Generate Examples
    pred_seqs = []
    label_seqs = []
    c_example = 0
    for  batch_data,batch_label in valid_set:
        
        if conf['model_parameter']['bucketing']:
            lstm_input_data = Variable(batch_data.squeeze(dim=0)).type(torch.FloatTensor).cuda()
            batch_label = batch_label.squeeze(0)
        else:
            lstm_input_data = Variable(batch_data).type(torch.FloatTensor).cuda()
        
        pred_seq = lstm(lstm_input_data)
    
        pred_seqs.append([char.cpu() for char in pred_seq])
        label_seqs.append([char.cpu() for char in batch_label])
        c_example += 1 
        if c_example > 10:
            break
 
    pred_seq_indexes = [np.argmax(s.detach().numpy(), axis=1) for b in pred_seqs for s in b]
    pred_seq_sentences = [ ''.join([idx2char[c] for c in s]) for s in pred_seq_indexes]
    label_seq_indexes = [np.argmax(s.detach().numpy(), axis=1) for b in label_seqs for s in b]
    label_seq_sentences = [ ''.join([idx2char[c] for c in s]) for s in label_seq_indexes]
 
    if global_step % 100 == 0:
        with open('pred_sentances' + str(global_step) + '.txt', 'w') as f:
            for i in range(len(pred_seq_sentences)):
                f.write(label_seq_sentences[i])
                f.write(' -- ')
                f.write(pred_seq_sentences[i])
                f.write('\n')
    
    #pd = {i:'' for i in range(conf['training_parameter']['batch_size'])}
    #for t, char in enumerate(pred_seq):
    #    print('t,char',t,char)
    #    for idx,i in enumerate(torch.max(char,dim=-1)[1]):
    #        print('idx,i',idx,i)
    #        print(' pd[idx]', pd[idx])
    #        if '<eos>' not in pd[idx]:
    #            pd[idx] += idx2char[int(i)]
    
    #pd = [pd[i] for i in range(conf['training_parameter']['batch_size'])]
   
    #gt = []
    #for line in (torch.max(batch_label,dim=-1)[1]).numpy():
    #    tmp = ''
    #    for idx in line:
    #        if idx == 0: continue
    #        if idx == 1: break
    #        tmp += idx2char[idx]
    #    gt.append(tmp)
    
    #for idx,(p,g) in enumerate(zip(pd,gt)):
    #    log_writer.add_text('pred_'+str(idx), p, global_step)
    #    if not record_gt_text:
    #        log_writer.add_text('test_'+str(idx), g, global_step)
   
    #if not record_gt_text:
    #    record_gt_text = True
    
     #att_map = {i:[] for i in range(conf['training_parameter']['batch_size'])}
     #num_head = len(attention_score[0])
     #for i in range(conf['training_parameter']['batch_size']):
     #    for j in range(num_head):
     #        att_map[i].append([])
     #for t,head_score in enumerate(attention_score):
     #    for h,att_score in enumerate(head_score):
     #        for idx,att in enumerate(att_score.data.numpy()):
     #            att_map[idx][h].append(att)
     #for i in range(conf['training_parameter']['batch_size']):
     #    for j in range(num_head):
     #        m = np.repeat(np.expand_dims(np.array(att_map[i][j]),0),3,axis=0)
     #        log_writer.add_image('attention_'+str(i)+'_head_'+str(j),
     #                             torch.FloatTensor(m[:,:len(pd[i]),:]), global_step)
    
    # Checkpoint
    if best_ler >= sum(val_ler)/len(val_ler):
        best_ler = sum(val_ler)/len(val_ler)
        print('Reached best CER',best_ler,'at step',global_step,',checkpoint saved.')
        torch.save(lstm, lstm_model_path)
        #torch.save(speller, speller_model_path)
