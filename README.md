Autonomous analysis of 1H anti-phase CEST experiments using DNNs
------------
------------
This code is for determining the chemical shifts of exchanging states from CEST experiments using DNN
To use the code, the file containing the weights for the trained network must be
downloaded. This is availble [Please provide link]
Once downloaded place the weights in a directory and point to that directory when running the programme

The script can then be run as follows:

./DNNautoCEST.py         \
   -sfrq SpecFreq        \
   -xcar CarrierFreq     \
   --datadir DATADIR     \
   --expname EXPNAME     \
   -model Directory_with_weights


The out, conf, gpu arguments are optional and will default to 'result', 0.4, and -1

Dependencies
------------
  * [Python=3.8.11](https://www.python.org/downloads/)
  * [Tensorflow=2.4.1](https://www.tensorflow.org/install)
  * [NumPy=1.20.3](https://www.scipy.org/scipylib/download.html)
  * [nmrglue=0.8](https://nmrglue.readthedocs.io/en/latest/install.html)
  * [matplotlib=3.4.2](https://matplotlib.org/stable/users/installing/index.html)  

  The script has been written and tested with the above dependencies.
  Performance with other module versions has not been tested.

  Please cite the following article if you use this script in your work:

  Karunanithy G., Yuwen T., Kay LE., Hansen DF.
  Towards autonomous analysis of Chemical Exchange Saturation Transfer experiments using Deep Neural Networks
  https://doi.org/10.26434/chemrxiv-2021-r1cmw
