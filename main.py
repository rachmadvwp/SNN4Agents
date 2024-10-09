from __future__ import print_function
import argparse
import numpy as np
from random import shuffle
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import os
import time
from spiking_model import*
from Ncars_dataset import*

PATH_RESULTS = './results'

#init value for python script
parser = argparse.ArgumentParser()
parser.add_argument('--filenet', type=str, dest='filename_net')
parser.add_argument('--fileresult', type=str, default='result.txt', dest='filename_result')
parser.add_argument('--sample_time', type=float, default=1, dest='sample_time')
parser.add_argument('--sample_length', type=float, default=10, dest='sample_length')
parser.add_argument('--batch_size', type=int, default=40, dest='batch_size')
parser.add_argument('--lr', type=float, default=1e-3, dest='lr')
parser.add_argument('--lr_decay_epoch', type=int, default=20, dest='lr_decay')
parser.add_argument('--lr_decay_value', type=float, default=0.5, dest='lr_decay_value')
parser.add_argument('--threshold', type=float, default=0.4, dest='thresh')
parser.add_argument('--n_decay', type=float, default=0.2, dest='n_decay') 
parser.add_argument('--att_window', type=int, nargs=4, dest='att_window')
parser.add_argument('--weight_decay', type=float, default=0, dest='weight_decay') #L2regularizzation
parser.add_argument('--wghbit_c0', type=int, default=32, dest='wghbit_c0')
parser.add_argument('--wghbit_c1', type=int, default=32, dest='wghbit_c1')
parser.add_argument('--wghbit_f0', type=int, default=32, dest='wghbit_f0')
parser.add_argument('--wghbit_f1', type=int, default=32, dest='wghbit_f1')
parser.add_argument('--quant', type=int, default=0, dest='quant') # 0: no quantization, 1: ptq
parser.add_argument('--tstep', type=int, default=20, dest='tstep') # timestep

args = parser.parse_args()

# initialize spiking model and network
initialize_model(args.filename_net, args.thresh, args.n_decay, 2, args.batch_size, args.lr, kernel_init_f=[args.att_window[0], args.att_window[1]])

batch_size = args.batch_size

data_path_train =  './'  
data_path_test =  './'   
data_path_results = './results/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

samplingTime = args.sample_time
sampleLength = args.sample_length
filename_result = args.filename_result
#
wghbitC0 = args.wghbit_c0
wghbitC1 = args.wghbit_c1
wghbitF0 = args.wghbit_f0
wghbitF1 = args.wghbit_f1
#
log_gpu = os.path.join(PATH_RESULTS, '_'.join(['log',str(args.att_window[0]),str(args.att_window[1])]) + '.txt')
with open(log_gpu, "w") as log: 
   log.write('log_'+str(args.att_window[0])+'_'+str(args.att_window[1])+'\n')
   log.write("===================================="+"\n") ###debug
log.close() 

# instantiate the train dataset and use the DataLoader function to give samples to the network 
trainingSet = DatasetHandler(datasetPath = data_path_train, 
									  sampleFile_car = './N_cars/car_train.txt',
									  sampleFile_background = './N_cars/background_train.txt',
									  samplingTime = samplingTime,
									  sampleLength = sampleLength,
									  shift_x = args.att_window[2],
 									  shift_y = args.att_window[3], 
									  att_window = [args.att_window[0],	args.att_window[1]])

train_loader = DataLoader(dataset=trainingSet, batch_size=batch_size, shuffle=True, num_workers=10)

# instantiate the test dataset and use the DataLoader function to give samples to the network 
testingSet = DatasetHandler(datasetPath = data_path_test, 
									 sampleFile_car = './N_cars/car_test.txt',
									 sampleFile_background = './N_cars/background_test.txt',
									 samplingTime = samplingTime,
									 sampleLength = sampleLength,
									 shift_x = args.att_window[2],
 									 shift_y = args.att_window[3], 
									 att_window = [args.att_window[0], args.att_window[1]])
test_loader = DataLoader(dataset=testingSet, batch_size=batch_size, shuffle=True, num_workers=10)

# create and open the file to write the results
file = os.path.join(data_path_results, filename_result+'.txt')
f = open(file, 'w')

# write the principal initialization information 
f.write('batch_size: '+str(args.batch_size)+ 
        ' sampling_time: '+str(samplingTime)+ 
        ' sampling_length: '+str(sampleLength)+ 
        ' filenet: '+str(args.filename_net)+ 
        ' learning_rate: '+str(args.lr)+ 
        ' lr decay_epoch: '+str(args.lr_decay)+ 
        ' lr decay_value: '+str(args.lr_decay_value)+ 
        ' threashold: '+str(args.thresh)+ 
        ' neuron_decay_constant: '+str(args.n_decay)+ 
        ' attention window: '+str(args.att_window)+ 
        ' weight_decay_(L2_reg): '+str(args.weight_decay)+
        ' weight_bit_conv0: '+str(wghbitC0)+
        ' weight_bit_conv1: '+str(wghbitC1)+
        ' weight_bit_fc0: '+str(wghbitF0)+
        ' weight_bit_fc1: '+str(wghbitF1)+'\n')

# define the network and load saved weights
snn = SCNN()
snn = putWeight(snn) # this part can be used to load the weigh of a previously trained network. 
snn.to(device)
#
# object for quantized model
snn_q = SCNN()

# define criterion and optimizer
criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(snn.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=False) #L2r

time_start = time.time()

# run the train and test for num_epochs epochs
for epoch in range(num_epochs):
   best_acc_entire_image_test = 0
   running_loss = 0
   start_time = time.time() 
   len_of_sample = len(trainingSet)

   # training ------------------------------------------------
   snn = snn.train()
   
   correct_entire_image = 0 # number of correct decision after sampleLngth/samplingTime predictions then choose the most predicted
   total_entire_image = 0   # number of images to predict
   #
   for i, (images, labels_, labels) in enumerate(train_loader,0):
      # run only for complete batches
      len_of_sample = len_of_sample-batch_size
      if len_of_sample >= 0:
         snn.zero_grad()
            
         optimizer.zero_grad()
         images = images.float().to(device)
         first = 0
	      # 
         # group outputs of the same image of length sampleLength and accumulate the prediction for every samplingTime
         for j in range (0, int(sampleLength/samplingTime)):
            outputs = snn(images[:,:,:,:,j], args)
            
            if first==0:
               _, accumulation = outputs.to(device).max(1)
               first = first+1
            else:
               _, predicted = outputs.max(1)
               accumulation += predicted
            #
            loss = criterion(outputs, labels_[:,:,0,0,0].to(device))
            running_loss += loss.item()
            loss.backward()
         optimizer.step()
         #
	      # see what is the most predicted class for the image
         accumulation[accumulation < (sampleLength/samplingTime)/2] = 0
         accumulation[accumulation >= (sampleLength/samplingTime)/2] = 1
         # 
	      # calculate accuracy on the image of length sampleLength
         total_entire_image += float(labels.size(0))
         correct_entire_image += float(accumulation.eq(labels.to(device)).sum().item())
         acc_entire_image_train = 100*correct_entire_image/total_entire_image
      #   
      if ((i+1)%20) == 0:
         print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f, Accuracy: %.5f' %(epoch+1, num_epochs, i+1, len(trainingSet)//batch_size, running_loss, acc_entire_image_train))
         running_loss = 0
         print('Time elasped:', time.time()-start_time)
	

   # testing ------------------------------------------------
   correct = 0 # number of correct decision for each samplingTime 
   total = 0 # number of total samplingTime predictions 
   optimizer = lr_scheduler(optimizer, epoch, args.lr_decay, args.lr_decay_value)
   correct_entire_image = 0 # number of correct decision after sampleLngth/samplingTime predictions then choose the most predicted
   total_entire_image = 0   # number of images of sampleLength length
   #
   with torch.no_grad():
      if (args.quant==1): # ptq
         # post-training quantization (PTQ)
         snn_q.conv[0].weight = torch.nn.Parameter(torch.floor(snn.conv[0].weight*(2**(args.wghbit_c0-1)))*(2**-(args.wghbit_c0-1)))
         print('snn_q.conv[0].weight: ', str(snn_q.conv[0].weight), '\n')
         #
         snn_q.conv[1].weight = torch.nn.Parameter(torch.floor(snn.conv[1].weight*(2**(args.wghbit_c1-1)))*(2**-(args.wghbit_c1-1)))
         print('snn_q.conv[1].weight: ', str(snn_q.conv[1].weight), '\n')
         #
         snn_q.fc[0].weight = torch.nn.Parameter(torch.floor(snn.fc[0].weight*(2**(args.wghbit_f0-1)))*(2**-(args.wghbit_f0-1)))
         print('snn_q.fc[0].weight: ', str(snn_q.fc[0].weight), '\n')
         #
         snn_q.fc[1].weight = torch.nn.Parameter(torch.floor(snn.fc[1].weight*(2**(args.wghbit_f1-1)))*(2**-(args.wghbit_f1-1)))
         print('snn_q.fc[1].weight: ', str(snn_q.fc[1].weight), '\n')
         #
         snn_q = snn_q.eval()
      else: # no quant 
         snn = snn.eval() 

      len_of_sample = len(testingSet)
      for batch_idx, (inputs, labels_, targets) in enumerate(test_loader,0):
         # run only for the complete batch size
         len_of_sample = len_of_sample-batch_size
         if len_of_sample >= 0:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            first = 0
	         # group outputs of the same image of length sampleLength and accumulate the prediction for every samplingTime
            for j in range (0, int(sampleLength/samplingTime)):
               if (args.quant==1 or args.quant==2): # ptq 
                  outputs = snn_q(inputs[:,:,:,:,j], args) 
               else: # no quant
                  outputs = snn(inputs[:,:,:,:,j], args) 
               #
               if first==0:
                  _, accumulation = outputs.to(device).max(1)
                  first = first+1
               else:
                  _, pre = outputs.max(1)
                  accumulation += pre
               #
               loss = criterion(outputs, labels_[:,:,0,0,0].to(device))
	            # calculate the prediction at every samplingTime without grouping them in an image of sampleLength length  
               _, predicted = outputs.max(1)
               total += float(targets.size(0))
               correct += float(predicted.eq(targets.to(device)).sum().item())
		         #
            # see the most predicted class for the image of length sampleLength
            accumulation[accumulation < (sampleLength/samplingTime)/2]=0
            accumulation[accumulation >= (sampleLength/samplingTime)/2]=1
	
	         # calculate accuracy on the image of length sampleLength and at every samplingTime
            total_entire_image += float(targets.size(0))
            correct_entire_image += float(accumulation.eq(targets.to(device)).sum().item())
            acc_entire_image_test = 100*correct_entire_image/total_entire_image
            #
            if (batch_idx%100)==0:
               acc = 100. * float(correct) / float(total)
               print(batch_idx, len(test_loader),' Acc: %.5f' % acc)

   print('Iters:', epoch,'\n\n\n')
   print('Test Accuracy of the model on the sampling time streams: %.3f' % (100 * correct / total))
   print('Test Accuracy of the model on the entire test images: %.3f' % (acc_entire_image_test))
   acc = 100. * float(correct) / float(total)
    
   # save the results at the every epoch  
   if epoch % 1 == 0:
      print(acc)
      print('Saving results..')
      #
      f.write('acc: '+str(acc)+' acc_train: '+str(acc_entire_image_train)+' acc_test: '+str(acc_entire_image_test)+' epoch: '+str(epoch)+'\n')
      #
      if (args.quant==1): # ptq 
         state = {'net': snn_q.state_dict(),
                  'acc': acc,
                  'epoch': epoch,}
      else: # no quant
         state = {'net': snn.state_dict(),
                'acc': acc,
                'epoch': epoch,}
      #
	   # save the network and the weights only if the accuracy on entire images is better than before 
      if epoch>=0 and best_acc_entire_image_test < acc_entire_image_test:
         print('Saving weights and network..')
         best_acc_entire_image_test=acc_entire_image_test
         if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
         torch.save(state, './checkpoint/ckpt' + str(args.att_window[0])+'_ceil' + '.t7')

time_end = time.time()
time_duration = time_start-time_end
### debug: start --------
with open(log_gpu, "a") as log: 
   log.write('Elapsed processing time: '+str(time_duration)+' seconds')
   log.write(" \n")
log.close() 
