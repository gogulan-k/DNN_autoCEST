#!/usr/bin/env python3

"""
  Written by D. Flemming Hansen, March-Dec 2021
  d.hansen@ucl.ac.uk

  When using this software, please cite: 

  Karunanithy G., Yuwen T., Kay LE., Hansen DF. 
  ChemRxiv (2021)
  https://doi.org/10.26434/chemrxiv-2021-r1cmw 

"""

######################################################
#
############ No need to change under here ############
#

import sys, os, copy
import argparse
parser = argparse.ArgumentParser(usage=' All required parameters have not been provided. \n \n Please run %s -h \n\n ' %(sys.argv[0]),description='CEST Analysis using DNNs *** \n Karunanithy G., Yuwen T., Kay LE., Hansen DF. \n https://doi.org/10.26434/chemrxiv-2021-r1cmw *** \n')

parser.add_argument('-sfrq','--sfrq',       help='Spectrometer frequency in MHz, e.g., 800.304 ', required=True)
parser.add_argument('-xcar','--xcar',       help='The carrier offset in ppm, e.g., 4.774 ', required=True)
parser.add_argument('-dir','--datadir',     help='Directory with cest profiles. All files within the directory are processed \n The format of each file must be [satfrq(Hz), Intensity(A.U.), Uncertainty(A.U.). \n The reference point needs to be provided with a saturation frequency less than 10,000 Hz \n', required=True)
parser.add_argument('-name','--expname',    help='The name of the dataset.', required=True)
parser.add_argument('-model','--modeldir',  help='Directory with the DNN weights ', required=True)
parser.add_argument('-out','--outname',     help='Name of output ', required=False, default='result')
parser.add_argument('-gpu','--gpu',         help='Which GPU to use. If -1 is provided then the CPU will be used', required=False, default="-1")
parser.add_argument('-conf','--conflimit',  help='The confidence level cut-off (see paper).', required=False, default='0.4' )

#
# Read in the arguments
args = vars(parser.parse_args())

SFRQ = float( args['sfrq'] )
XCAR = float( args['xcar'] )
ExpDataDirs=[ args['datadir']+"/" ]
ExpNames = [ args['expname'] ]
BaseDir= args['modeldir']+"/"
if int(args['gpu'])<0:
    print(' INFO: Processing using only CPU ')
    os.environ['CUDA_VISIBLE_DEVICES']=""
else:
    print(' INFO: Processing with GPU number %d ' %(int(args['gpu'])))
    os.environ['CUDA_VISIBLE_DEVICES']="%d" %(int(args['gpu']))

#
# Set required parameters for this model
UpSample=True
UpSampleTD=128
OnlyEven=True
NP=65

TimeSize = NP

import time
sys.stderr.write('%-35s' %(' Reading libraries ... '))
sys.stderr.flush()
start_time=time.time()

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow        as tf
from   tensorflow import keras
import numpy             as np
import matplotlib.pyplot as plt
import nmrglue           as ng
from   matplotlib.backends.backend_pdf import PdfPages
import tensorflow.keras.backend as K
#
# Turn warnings off
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

sys.stderr.write('\033[1;32m done in %.1f seconds \033[0;39m \n' %(time.time()-start_time))

#########################################

ParamFile=BaseDir +'cest_ipap.h5'

sys.stderr.write('%-35s' %(' Prepare input profiles  ... '))
sys.stderr.flush()
start_time=time.time()

#
# Define functions for peak-picking
def build_model_denseConv(filters):

    def dense_block(x, filty, num_layers):
        stack = []
        x = alt_conv(x, filty)
        stack.append(x)

        for k in range(num_layers-1):
            if k>0:
                x = alt_conv(keras.layers.Concatenate()(stack), filty)
            else:
                x = alt_conv(x, filty)

            stack.append(x)

        return keras.layers.Concatenate()(stack)

    def alt_conv(x, filty):
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv1D(4*filty, kernel_size=1, strides=1, padding = 'same')(x)

        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv1D(filty, kernel_size=3, strides=1, padding = 'same')(x)
        return x

    def transition(x):
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.AveragePooling1D(pool_size=2, strides = 2)(x)

        return x

    input =  keras.layers.Input(shape=[128,1])
    x = input
    x = keras.layers.Conv1D(filters, kernel_size=7, strides=1, padding = 'same')(x)
    x = keras.layers.MaxPool1D(pool_size=3, strides=2)(x)

    x = dense_block(x, filters, 6)
    x = transition(x)

    x = dense_block(x, filters, 12)
    x = transition(x)

    x = dense_block(x, filters, 32)
    x = transition(x)

    x = dense_block(x, filters, 32)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    fin_dens = keras.layers.Dense(6, activation="sigmoid")(x)
    fin_dens_reshape = keras.layers.Reshape((3, 2), input_shape=(6,))(fin_dens)

    model = keras.Model(inputs=[input], outputs=[fin_dens_reshape])

    model.compile(loss=unique_pairs_loss,
                    optimizer=keras.optimizers.Nadam(lr=1.0e-4),
                    run_eagerly=False)

    return model

def unique_pairs_loss(yTrue, yPred):
    return 0.

def conf_loss(yTrue, yPred):
    return 0.
###

def get_cest_from_int(l):
    for i in range(len(l)):
        if float(l[i][0])<-1.e4:
            ref_id=i
    ref = l.pop( ref_id )
    #
    retval = np.array(l)
    retval[:,1] = retval[:,1]/ref[1]
    if OnlyEven:
        if not retval.shape[0] % 2 == 0 :
            retval = retval[ : 2*(retval.shape[0]//2),:]
    #
    return retval
#
# Read in data
profiles={}
profiles['params']=[]
input_ipaps=[]
#
for ec in range(len(ExpDataDirs)):
    exp_profiles = []
    profiles['params'].append({})
    profiles['params'][-1]['names']=[]
    for filename in os.listdir(ExpDataDirs[ec]):
        if filename.endswith('.out'):
            profiles['params'][-1]['names'].append(filename)
            data=[]
            ifs=open(ExpDataDirs[ec]+filename,'r')
            for l in ifs.readlines():
                data.append( [float(k) for k in l.split()] )
            cest_profile = get_cest_from_int(data)
            #
            # only take 1/2 of the points
            #cest_profile = cest_profile[0::2,:]
            exp_profiles.append( cest_profile )
    #
    exp_profiles = np.array( exp_profiles )
    #
    # This is where we do some testing
    profiles['ipap_%d' %(ec,)] = np.copy( exp_profiles[:,:,0:2] )
    #
    # Scale to maximum of 0.90
    profiles['params'][-1]['Scaling'] = np.max( np.vstack([ 1.00*np.ones(exp_profiles.shape[0]), 1.1*np.max( np.fabs( exp_profiles[:,:,1]), axis=-1)]), axis=0)
    for r in range(profiles['ipap_%d' %(ec,)].shape[0]):
        if profiles['params'][-1]['Scaling'][r] > 1.0:
            sys.stderr.write('\n WARNING: Scaling needed for residue %s in experiment %s ' %(profiles['params'][ec]['names'][r].split('.')[0],ExpNames[ec]))
        profiles['ipap_%d' % (ec,)][r,:,1] = -profiles['ipap_%d' %(ec,)][r,::-1,1]/profiles['params'][-1]['Scaling'][r]
    profiles['params'][-1]['TD'] = exp_profiles.shape[1]
    profiles['params'][-1]['SW'] = np.max( exp_profiles[0,:,0] ) - np.min( exp_profiles[0,:,0])
    profiles['params'][-1]['LEFT'] = np.max( exp_profiles[0,:,0] )
    profiles['params'][-1]['RIGHT']= np.min( exp_profiles[0,:,0] )
    profiles['params'][-1]['NRes'] = exp_profiles.shape[0]
    profiles['params'][-1]['SFRQ'] = SFRQ
    profiles['params'][-1]['XCAR'] = XCAR
    
    #profiles['params'] = np.array( profiles['params'])
    NRes = profiles['params'][-1]['NRes']
    #
    # Do real FT
    ipap_ft = np.zeros( ( profiles['ipap_%d' %(ec,)].shape[0], NP ,3), dtype=np.float32 )
    input_ipap = np.zeros( (profiles['ipap_%d' %(ec,)].shape[0], NP*2 ), dtype=np.float32 )
    #target     = np.zeros( (profiles['ipap'].shape[0], NP*2 ), dtype=np.float32 )
    #
    import matplotlib.pyplot as plt
    for r in range(profiles['ipap_%d' %(ec,)].shape[0]):
        this_ft = 0.5*np.fft.rfft( profiles['ipap_%d' %(ec,)][r,:,1], axis=-1 )/profiles['params'][-1]['TD']
        ipap_ft[r,:profiles['params'][-1]['TD']//2+1,1] = np.real( this_ft )
        ipap_ft[r,:profiles['params'][-1]['TD']//2+1,2] = np.imag( this_ft )    
        ipap_ft[r,:,0] = np.copy(np.arange( NP )/profiles['params'][-1]['SW'])
    #
    for r in range(profiles['ipap_%d' % (ec,)].shape[0]):
        input_ipap[r,: ] = tf.reshape( ipap_ft[r,:,1:3] , shape = (NP*2,)).numpy()
    input_ipaps.append( np.copy( input_ipap ))

    
profiles['params'] = np.array( profiles['params'] )
#
TF_input_ipap= tf.concat( [ tf.constant( input_ipaps[i], dtype=tf.dtypes.float32 ) for i in range(profiles['params'].shape[0] )], axis=0)

time_input = np.zeros( (TF_input_ipap.shape[0],2*NP)  , dtype=np.float32 )
tc=0
for ec in range( profiles['params'].shape[0] ):
    for r in range( profiles['params'][ec]['NRes']):
        time_input[tc,0::2] = np.copy(np.arange( NP )/profiles['params'][ec]['SW'])
        time_input[tc,1::2] = np.copy(np.arange( NP )/profiles['params'][ec]['SW'])
        tc+=1
        
TF_input_time = tf.constant( time_input, dtype=tf.dtypes.float32 )

sys.stderr.write('\033[1;32m done in %.1f seconds \033[0;39m \n' %(time.time()-start_time))
sys.stderr.write('%-35s' %(' Reading DNN model  ... '))
sys.stderr.flush()
start_time=time.time()

# set up the model
ModelFile= BaseDir + 'model_cest.json'
model_json=open(ModelFile,'r').read()

def dfh_gelu(x):
    return tf.add(tf.tanh(x), tf.math.scalar_mul( tf.constant(0.02, dtype=tf.dtypes.float32) ,x))
def dfh_sigm(x):
    return tf.add(tf.math.sigmoid(x), tf.math.scalar_mul( tf.constant(0.02, dtype=tf.dtypes.float32), x))

model=keras.models.model_from_json(model_json, custom_objects={'dfh_gelu': tf.keras.layers.Activation(dfh_gelu), 'LeakyReLU': tf.keras.layers.LeakyReLU(alpha=0.2), 'dfh_sigm': tf.keras.layers.Activation(dfh_sigm)} )

model.load_weights(ParamFile)

sys.stderr.write('\033[1;32m done in %.1f seconds \033[0;39m \n' %(time.time()-start_time))
sys.stderr.flush()

sys.stderr.write('%-35s' %(' Converting CEST profiles ... '))
sys.stderr.flush()
start_time=time.time()

out_time=model.predict( (TF_input_time[:,:], TF_input_ipap[:,:] ) )

sys.stderr.write('\033[1;32m done in %.1f seconds \033[0;39m \n' %(time.time()-start_time))
sys.stderr.write('%-35s' %(' Peak-pick and plotting ... '))
sys.stderr.flush()
start_time=time.time()

data_to_save = []

#
# Setup peak-picking
model_pp = keras.models.load_model(BaseDir+'./cest_denseConv_128.h5', custom_objects={'unique_pairs_loss': unique_pairs_loss, 'conf_loss': conf_loss})

def conf2sigma(conf, minppm=6.33, maxppm=9.55):
    return np.fabs(minppm- maxppm)*np.sqrt(6.e-5)*(1./conf - 1.)

ofs=open(args['outname']+'.out','w')
ofs.write('# Results in brackets () have confidence level < %.3f \n' %(float(args['conflimit'])))
ofs.write('#%19s %14s ' %('ExpName','ResName'))
for i in range(3):
    ofs.write(' %7s %5s %5s  ' %('Pos(%d)' %(i,), 'Conf','Std'))    
ofs.write('\n')
ofs.write('#                                     -----  [ppm] -----    -----  [ppm] -----    -----  [ppm] ----- \n')
    
with PdfPages(args['outname']+'.pdf') as pdf:
    for ec in range(profiles['params'].shape[0]):
        if ( (ec+1) * NRes > out_time.shape[0] ): break
        TD = profiles['params'][ec]['TD']
        SFRQ=profiles['params'][ec]['SFRQ']
        #
        # Let's get the CEST curves!
        ipap_frq  = np.fft.irfft( (input_ipaps[ec][:,0::2] + input_ipaps[ec][:,1::2]*1j)[:,:TD//2+1], axis=-1 )*TD
        if UpSample:
            out_frq   = np.fft.irfft( (out_time[ec*NRes:(ec+1)*NRes,0::2] + out_time[ec*NRes:(ec+1)*NRes,1::2]*1j)[:,:UpSampleTD//2+1], axis=-1 )*UpSampleTD/NP
        else:
            out_frq   = np.fft.irfft( (out_time[ec*NRes:(ec+1)*NRes,0::2] + out_time[ec*NRes:(ec+1)*NRes,1::2]*1j)[:,:TD//2+1], axis=-1 )*TD/NP
        #        
        offsets  = np.linspace( profiles['params'][ec]['RIGHT']/profiles['params'][ec]['SFRQ'] + profiles['params'][ec]['XCAR'], \
                                profiles['params'][ec]['LEFT']/profiles['params'][ec]['SFRQ'] + profiles['params'][ec]['XCAR'], \
                                TD, endpoint=True )
        offsets_full = np.linspace( profiles['params'][ec]['RIGHT']/profiles['params'][ec]['SFRQ'] + profiles['params'][ec]['XCAR'], \
                                profiles['params'][ec]['LEFT']/profiles['params'][ec]['SFRQ'] + profiles['params'][ec]['XCAR'], \
                                UpSampleTD, endpoint=True )

        #
        # Store ppm range
        minppm = profiles['params'][ec]['RIGHT']/profiles['params'][ec]['SFRQ'] + profiles['params'][ec]['XCAR']
        maxppm = profiles['params'][ec]['LEFT']/profiles['params'][ec]['SFRQ'] + profiles['params'][ec]['XCAR']
        #                       
        for i,r in enumerate(np.argsort( profiles['params'][ec]['names'])):
            if i==0:
                if int(args['gpu'])<0:
                    sys.stderr.write('\r%-35s profile %3d/%d  (loading weights to mem) ' %(' Peak-pick and plotting ... ',i+1,out_frq.shape[0]))                    
                else:
                    sys.stderr.write('\r%-35s profile %3d/%d  (loading weights to GPU) ' %(' Peak-pick and plotting ... ',i+1,out_frq.shape[0]))                    
            else:
                sys.stderr.write('\r%-35s profile %3d/%d                           ' %(' Peak-pick and plotting ... ',i+1,out_frq.shape[0]))
            sys.stderr.flush()
            #
            HaveCS=False
            try:
                CS = profiles['params'][ec]['CS'][r]
                HaveCS=True
            except(KeyError):
                HaveCS=False
            
            plt.figure(figsize=(10,6))
            ax2 = plt.subplot2grid( (2,1), (0,0 ) )
            ax1 = plt.subplot2grid( (2,1), (1,0 ) )
            #
            try:
                if UpSample:
                    ax1.plot( offsets_full, np.max(2.*out_frq[r,:])-2*out_frq[r,:], 'c-', label='DNN (IP)' )
                    data_to_save.append( [offsets_full,  2.*out_frq[r,:] ] )
                #
                if(HaveCS):
                    ax1.plot( [CS[0]-0.5*93./SFRQ,CS[0]-0.5*93./SFRQ], [-0.1,1], 'k--', label=r'$\delta_A$')
                    ax1.plot( [CS[1]-0.5*93./SFRQ,CS[1]-0.5*93./SFRQ], [-0.1,1], '--', color='gray', label=r'$\delta_B$' )
                #
                # Now peak-pick
                cest_ft_pp = tf.reshape( tf.constant(2.*out_frq[r,:], dtype=tf.dtypes.float32), (1,-1,1))
                norm_factor_pp = tf.reduce_max(cest_ft_pp)
                cest_ft_pp = cest_ft_pp/norm_factor_pp
                #
                res_pp = model_pp.predict(cest_ft_pp)
                #
                locs = 1.10*res_pp[0,:,0]-0.05
                probs = res_pp[0,:,1]
                locs = locs*(np.max(offsets_full)-np.min(offsets_full))+ np.min(offsets_full)
                cest_int = 2.*np.ones(3)
                for i in range(3):
                    if probs[i]>float(args['conflimit']):
                        #
                        # Is this the ground state?
                        # Where are we?
                        best_pts = int(np.interp(locs[i],offsets_full, np.arange(len(offsets_full))))
                        cest_int[i] = np.max(2.*out_frq[r,:]) - 2.*out_frq[r,best_pts]
                sorting= np.argsort(cest_int)
                #
                # check for confidence
                if probs[sorting[2]]>probs[sorting[1]]:
                    # Swap
                    _tt = sorting[2]
                    sorting[2] = sorting[1]
                    sorting[1] = _tt
                colours=['b','r','k']
                for i in range(3):
                    if probs[sorting[i]]>float(args['conflimit']):
                        ax1.plot([locs[sorting[i]],locs[sorting[i]]],[0.0,0.82+0.05*i], colours[i]+'-')
                        ax1.plot([locs[sorting[i]]+conf2sigma(probs[sorting[i]],minppm=minppm, maxppm=maxppm),
                                  locs[sorting[i]]+conf2sigma(probs[sorting[i]],minppm=minppm, maxppm=maxppm)], [0.0,0.82+0.05*i], colours[i]+'--')
                        ax1.plot([locs[sorting[i]]-conf2sigma(probs[sorting[i]],minppm=minppm, maxppm=maxppm),
                                  locs[sorting[i]]-conf2sigma(probs[sorting[i]],minppm=minppm, maxppm=maxppm)], [0.0,0.82+0.05*i], colours[i]+'--')
                        
                        ax1.text(locs[sorting[i]],0.85+0.05*i, str(conf2sigma(probs[sorting[i]], minppm=minppm, maxppm=maxppm))[0:5], ha = 'left')
                #sys.exit(10)
                ax2.plot( offsets, profiles['ipap_%d' %(ec,)][r,:,1], 'm.-', label='orig' )
                if(HaveCS):
                    ax2.plot( [CS[0],CS[0]], [0,1], 'k--', label=r'$\delta_A$')
                    ax2.plot( [CS[1],CS[1]], [0,1], '--', color='gray', label=r'$\delta_B$' )
                #
                ax1.legend(frameon=False)
                ax2.legend(frameon=False)
                ax2.set_ylabel(r'${\rm IPAP\ CEST\ profile } \quad I/I_{0} $')
                ax1.set_xlabel(r'$B_1 \ {\rm offset } \quad {\rm (ppm) }$')
                #
                ax1.set_ylabel(r'${\rm IP\ Transformed\ CEST } \quad I/I_{0} $')
                #
                ax1.set_ylim( (-0.05,1.05) )
                ax2.set_ylim( (-1.05,1.05) )
                ax1.set_xlim( (np.min(offsets_full),np.max(offsets_full) ) )
                ax2.set_xlim( (np.min(offsets_full),np.max(offsets_full) ) )
                ax1.invert_xaxis()
                ax2.invert_xaxis()
                #
                ax2.set_xticks([])
                #
                plt.suptitle('Data from: %s (TD=%d), residue: %s' %( ExpNames[ec], profiles['params'][ec]['TD'], profiles['params'][ec]['names'][r] ) )
                ofs.write('%20s %14s ' %(ExpNames[ec], profiles['params'][ec]['names'][r]))
                for i in range(3):
                    if probs[sorting[i]]>float(args['conflimit']):
                        ofs.write(' %7.3f %5.3f %5.3f  ' %(locs[sorting[i]],probs[sorting[i]], conf2sigma(probs[sorting[i]],minppm=minppm, maxppm=maxppm)))
                    else:
                        ofs.write('(%7.3f %5.3f %5.3f) ' %(locs[sorting[i]],probs[sorting[i]], conf2sigma(probs[sorting[i]],minppm=minppm, maxppm=maxppm)))                        
                ofs.write('\n')
                #
                #
                pdf.savefig()
                plt.close()

            except KeyboardInterrupt:
                print(' Caught ctrl-c; quitting ', file=sys.stderr)
                plt.clf()
                sys.exit(10)

        data_to_save=[]

sys.stderr.write('\r%-35s' %(' Peak-picking and plotting ... '))            
sys.stderr.write('\033[1;32m done in %.1f seconds \033[0;39m                  \n' %(time.time()-start_time))
ofs.close()
sys.stderr.flush()
