# CERN ATLAS Autoencoders 🔬

**＊ ✿ ❀ Training an autoencoder to compress data from CERN's high energy physics dataset ❀ ✿ ＊**


<div>
  
  [![Status](https://img.shields.io/badge/status-active-success.svg)]()
  [![GitHub Issues](https://img.shields.io/github/issues/lucylow/CERN_HEP_Autoencoder.svg)](https://github.com/lucylow/CERN_HEP_Autoencoder/issues)
  [![GitHub Pull Requests](https://img.shields.io/github/issues-pr/lucylow/CERN_HEP_Autoencoder.svg)](https://github.com/lucylow/CERN_HEP_Autoencoder/pulls)
  [![License](https://img.shields.io/aur/license/android-studio.svg)]()

</div>


---

## Table_of_Contents &#x1F49C;

* [Motivation](#Motivation-)
* [Autoencoders](#Autoencoders-)
* [Download the HEP data](#Download_the_HEP_data-)
* [Run the training script](#Run_the_training_script-) 
* [Loss error function](#Loss_error_function-)
* [TensorBoard monitoring model training](#TensorBoard_monitoring_model_training-)
* [Conclusion Model Discussion](#Conclusion_Model_Discussion-)
* [References](#references-) 

---

## Motivation &#x1F49C;

* Train an autoencoder(AE) to compress the files in the dataset from 4 to 3 variables. https://drive.google.com/drive/folders/1JaCB-prsDhEX4Ovk-UjC9bMxOHbpfREr?usp=sharing

Lossy compression of hadron jet data using autoencoders (AE)

Our goal: train Neural Networks to compress event size even further • Train using large amounts of
data
• Use for fast compression
online


* Technical Requirements 
  * PyTorch
  * FastAI Library https://docs.fast.ai/install.html
  * ROOT Data Analysis Framework 
  
Setup your ML environment with CERN ATLAS and docker containters. Contains dockerfiles for images that contain ATLAS and ML components with atlas-sit/docker as a dependency

https://gitlab.cern.ch/aml/containers/docker


---

## Autoencoders &#x1F49C;
* "Autoencoding" == **Data compression algorithm** with compression and decompression functions
* User defines the parameters in the function using variational autoencoder
* Self-supervised learning where target models are generated from input data
* Implemented with **neural networks** - useful for problems in unsupervised learning (no labels)

---

## Variational Autoencoders (VAE) &#x1F49C;

* Variational autoencoders are autoencoders, but with more constraints
* **Generative model** with parameters of a probability distribution modeling the data
* The encoder, decoder, and VAE are 3 models that share weights. After training the VAE model, **the encoder can be used to generate latent vectors**

Example of **encoder network maping inputs to latent vectors**:

* Input samples x into two parameters in latent space = **z_mean and z_log_sigma** 
* Randomly sample points z from latent normal distribution to generate data
* z = z_mean + exp(z_log_sigma) * epsilon, where epsilon is a **random normal tensor**
* **Decoder network maps latent space** points back to the original input data

```python

 x = Input(batch_shape=(batch_size, original_dim))
 
 h = Dense(intermediate_dim, activation='relu')(x)
 
 z_mean = Dense(latent_dim)(h)
 
 z_log_sigma = Dense(latent_dim)(h)
```
*Sample Code for VAE encoder network*

---
  
## Prepare_the_node_environment &#x1F49C;


```sh
yarn
# Or
npm install
```

---

## Download_the_HEP_data &#x1F49C;
* Download dataset https://drive.google.com/drive/folders/1JaCB-prsDhEX4Ovk-UjC9bMxOHbpfREr
* Uncompress the large file size (26 MBytes)
* Move files into `dataset` folder in the example folder of this repo
* Data is in a pickle python format: 

> import pandas

> object = pd.read_pickle(r'filepath')


---

## Run_the_training_script &#x1F49C;

* For this project, we don't need to train the network since there is an ready-made network in the folder. However, there is a training/testing dataset if you want to train your own model:

* Can not feed all the data to the model at once due to computer memory limitations so **data is split into "batches"** 
* When all batches are fed exactly once, an "epoch" is completed. As training script runs, **preview images afer every epoch will show**

```sh
yarn train
```
---

## Performance Plot Comparison  &#x1F49C;

Produce plots of the difference between original and uncompressed variables for each entry of the dataset, divided by the original variable. You can also add other plots (eg reconstruction loss).

does the network work well for compression based on them?


---

## Loss_error_function &#x1F49C;

* **Loss function to account for error in training** since Ms.Robot is picky about her fashion pieces 
* Two loss function options: The default **binary cross entropy (BCE)** or **mean squared error (MSE)**
* The loss from a good training run will be approx 40-50 range whereas an average training run will be close to zero

  ![Example loss curve from training](https://github.com/lucylow/Ms.Robot/blob/master/images/vae_tensorboard2.png)

    *Image of loss curve with the binary cross entropy error function*


---

## Conclusion_Model_Discussion &#x1F49C;

Results show a _________ model with parameters of a probability distribution variational autoencoder (VAE) is capable of achieving results on a highly challenging dataset of over ____________ images using machine learning. This is done by scanning the latent plane, sampling the latent points at regular intervals, to generate the corresponding ____________ for each point. 

AEs have been shown to successfully compress and reconstruct simple jet data

From preliminary tests on data and signal, the quality of the reconstruction would
be good enough for applications such as TLA (where a high precision is not
required as we already use trigger jets for the search)

Future experimental improvements includes the discrete variables could be treated more appropriately, e.g. using one-hot
encoding

Explore other architectures, e.g. using skipping connections, auxiliary inputs,
multiple branches, more/fewer nodes, more/fewer layers

Compress event-by-event instead of jet-by-jet, e.g. using recurrent neural networks
(RNNs)
Performance was successfully evaluated on a MC signal sample not seen by the AE
during training

HOW WOULD YOU IMPROVE THIS NETWORK?? Where else could you think of using a similar network?



Future questions

Physics questions (from the TLA workshop)
• Can we show that this compression algorithm will not create a significant
number of spurious jets / make spurious jets look normal jets?
• What is the interplay with the use of autoencoders for compression and for
anomaly detection?
• Can we ”chain” this kind of compression to other lossless/lossy compression
algorithms we are using in ATLAS?
• What are the CPU/timing costs of such a compression?
Retrain a similar network on gravitational wave data (time series, difficult to
compress)
Far future steps
Understand how to use this kind of network for compression in jets/events in the
bytestream
Write a PUB note?
Any further suggestions?



---

## References &#x1F49C;
* Unpack the Jet dataset: https://stackoverflow.com/questions/24906126/how-to-unpack-pkl-file
* ML compression of ATLAS triggered jet events using autoencoders https://github.com/Skelpdar/HEPAutoencoders
* Eric Wulff. Deep Autoencoders for Compression in High Energy Physics Paper https://lup.lub.lu.se/student-papers/search/publication/9004751 and https://indico.cern.ch/event/870013/contributions/3668675/attachments/1960856/3260728/Autoencoders_for_jet_compresion_in_HEP.pdf 
* FastAI Autoencoders https://alanbertl.com/autoencoder-with-fast-ai/
* Accessing a ROOT TTree With a TBrowser https://root.cern.ch/tutorials
* New Approaches to High Level Particle Physics Analysis https://indico.ph.ed.ac.uk/event/66/contributions/832/attachments/698/853/200215__Analysis_2.0__Python__HEP__...3.pdf
* FAST HEP Helping turn your trees into tables (ie. reads ROOT TTrees, writes summary Pandas DataFrames) https://github.com/fast-hep/fast-carpenter
* Demo analysis based on the CMS Public HEP Tutorial from 2012 https://gitlab.cern.ch/fast-hep/public/fast_cms_public_tutorial
* Docker FAST HEP https://hub.docker.com/r/fasthep/fast-hep-docker
