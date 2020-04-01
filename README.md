# Deep Neural Network Autoencoders for Data Compression in High Energy Physics 🔬&#x1F499;

**＊ ✿ ❀ Use an autoencoder to compress hadron jet event data from 4 to 3 variables from CERN's high energy physics dataset ❀ ✿ ＊**

<div>
  
  [![Status](https://img.shields.io/badge/status-active-success.svg)]()
  [![GitHub Issues](https://img.shields.io/github/issues/lucylow/CERN_HEP_Autoencoder.svg)](https://github.com/lucylow/CERN_HEP_Autoencoder/issues)
  [![GitHub Pull Requests](https://img.shields.io/github/issues-pr/lucylow/CERN_HEP_Autoencoder.svg)](https://github.com/lucylow/CERN_HEP_Autoencoder/pulls)
  [![License](https://img.shields.io/aur/license/android-studio.svg)]()

</div>

---

## Table_of_Contents;

* [Motivation](#Motivation-)
* [High_Energy_Physics](#High_Energy_Physics-)
* [CERN's_Large_Hadron_Collider](#CERN's_Large_Hadron_Collider-)
* [ATLAS](#ATLAS-) 
* [ATLAS_Particle_Physics_Trigger_System](#ATLAS_Particle_Physics_Trigger_System-)
* [Machine_Learning_Autoencoders](#Machine_Learning_Autoencoders-)
* [Technical_Data_Analysis](#Technical_Data_Analysis-)
* [CERN's_Large_Hadron_Collider](#CERN's_Large_Hadron_Collider-)
* [Setup_Docker_Environment](#Setup_Docker_Environment-) 
* [Performance_Analysis_Plot_Comparison](#Performance_Analysis_Plot_Comparison-)
* [Future_Experimental_Suggestions](#Future_Experimental_Suggestions-) 
* [References](#references-) 

---

## Motivation;
* Physicists belonging to worldwide collaborations work continuously to improve **machine learning methods in high energy physics** to detect ever more interesting events. Their goal is to understand more of the subatomic world and the laws of physics that govern them and answer some of the fundamental open questions in science. Discovery of the Higgs boson has opened up whole new windows in the search for new physics, to **search for evidence of theories of particle physics beyond the Standard Model**. This requires high energy collisions and particle decays after very short periods of time.
* Storage is one of the main limiting factors to the recording of information from proton-proton collision events at the Large Hadron Collider. Scientists want to reduce the size of the data that is recorded, and **study compression algorithms that can be used directly within the trigger system**. Gain expertise in cutting-edge machine learning techniques, and learn to use them in the context of **data compression and detection of anomalous events**. 
* In this experiment, an **autoencoder (AE) is used to compress hadron jet event data from 4 to 3 variables** (Located in /dataset). **That analysis is in the Juypter Notebook: https://github.com/lucylow/CERN_HEP_Autoencoder/blob/master/autoencoder.ipynb** 

* **Technical Requirements:**
  * PyTorch
  * FastAI Library https://docs.fast.ai/install.html
  * ROOT Data Analysis Framework https://root.cern.ch/
  * ML environment Dockerfiles for images that contain CERN ATLAS and ML components (https://gitlab.cern.ch/aml/containers/docker)

---
## High_Energy_Physics;
* **Standard Model of elementary particles**
  * The Higgs boson
* **Sub-atomic particles** 
  * Bosons 
  * Hadrons
  * Fermions
* **Hadrons** 
  * Subatomic composite particle made of two or more quarks held together by the strong force in a similar way as molecules are held together by the electromagnetic force
  * Two types of Hardons:
    * Baryons Ex. Protons and neutrons 
    * Mesons Ex. Pions
  * Like all subatomic particles, **hadrons are assigned quantum numbers** corresponding to the representations of the **JPC(m) group**: 
    * J is the spin quantum number
    * P the intrinsic parity (or P-parity)
    * C the charge conjugation (or C-parity)
    * m the particle's mass


![dsada](https://github.com/lucylow/CERN_HEP_Autoencoder/blob/master/images/Screen%20Shot%202020-03-11%20at%201.40.19%20PM.png)


---
## CERN's_Large_Hadron_Collider;
* **Large Hadron Collider (LHC)** == world's largest and highest-energy particle collider built by European Organization for Nuclear Research (CERN) in Geneva 
*  British scientist Tim Berners-Lee invented the **World Wide Web (WWW) in 1989, while working at CERN** originally to meet the demand for automated information-sharing between scientists around the world
* Dectectors allow physicists to **test the predictions of different theories of particle physics, including measuring the properties of the Higgs boson** 
* Collider is a type of particle accelerator with two directed beams of particles. It is used as an experimental research tool to **accelerate particles to high kinetic energies** and let them impact other particles

----

## ATLAS;
  * ATLAS is the **largest general-purpose particle detector experiment** at the Large Hadron Collider
  * Experiment was designed to measure the broadest possible range of signals. It is designed to detect these particles, namely their masses, momentum, energies, lifetime, charges, and nuclear spins
  * In July 2012, it was involved in the discovery of the Higgs boson. Higgs mechanism is essential to explain the generation mechanism of the property "mass" for gauge bosons. P**eter Higgs and François Englert had been awarded the 2013 Nobel Prize in Physics** after serach found Higgs boson. Yay!
* Generates large amounts of data ~ total of 1 petabyte of raw data per second. 25 megabytes per event (raw; zero suppression reduces this to 1.6 MB), multiplied by 40 million beam crossings per second in the center of the detector
* The remaining data, corresponding to about 1000 events per second, are stored for further analysis
  
![dsada](https://github.com/lucylow/CERN_HEP_Autoencoder/blob/master/images/Screen%20Shot%202020-03-11%20at%201.41.16%20PM.png)  


---


## ATLAS_Particle_Physics_Trigger_System;
* Particle Physics Trigger System (https://atlas.cern/discover/detector/trigger-daq)
* 40 million packets of protons collide every second at the centre of the ATLAS detector during LHC operation. Due to the extremely high LHC collision rate of up to 20 MHz not all events can be stored. 
* **Particle Physics Trigger System** == A trigger system selects specific events and writes them to disk for further analysis.System decides which events in a particle detector to keep when only a small fraction of the total can be recorded. A small subset of these collisions are passed through visualisation software and displayed on a large screen in the ATLAS Control Room 
* **Selectivity of the trigger:**
  * The ratio of the trigger rate to the event rate   
  * LHC has an event rate of 40 MHz (4·107 Hz), and the Higgs boson is expected to be produced there at a rate of roughly 1 Hz. 
* **Two trigger levels**
  * Level 1 : Information from the calorimeters and the muon spectrometer, and decreases the rate of events in the read-out to 100 kHz
  * Level 2 : Limited regions of the detector, so-called regions of interest (RoI), to reconstruct events by matching energy deposits to tracks



---

## Machine_Learning_Autoencoders;

* Autoencoder (AE) netural networks commonly used for compression and anomaly detection
* **AEs have been shown to successfully compress and reconstruct simple jet data.** Data compression algorithm with compression and decompression functions
* Implement an approximation to the identity, f(x) ≈ x, by using one or more hidden layers with smaller size than the input and output layers such that the information necessary to reproduce the input, x, is contained in the hidden layer, and the data has been compressed. This smaller hidden layer representation is saved instead of the current data format, along with the neural network that can recreate the original data.
* For **anomaly detection, the AE is first trained on data which is known not to be anomalous**. If then the network is presented with a new data point that differs in some significant way from the training data, the AE will not be able to provide a faithful reconstruction at the output layer and hence the data point is considered anomalous where if the reconstruction error of a data point is larger than some threshold, it can be classified as anomalous.
* ROOT framework in the HEP community paper **"Exploring compression techniques for ROOT IO"** https://arxiv.org/abs/1704.06976

---
## Technical_Data_Analysis;
* Download jet dataset https://drive.google.com/drive/folders/1JaCB-prsDhEX4Ovk-UjC9bMxOHbpfREr
  * Data is in a pickled python format: 
  > import pandas
  > object = pd.read_pickle(r'filepath')

* **The jet event information for four-momentum components:**
  * Leading jets
  * Electrons
  * Muons
  * Photons

![dsada](https://github.com/lucylow/CERN_HEP_Autoencoder/blob/master/images/Screen%20Shot%202020-03-11%20at%208.11.20%20PM.png)  


* Introduction to the analysis framework, including examples for producing histograms of basic quantities such as momentum distributions
* Using plots and graphical analysis to explain the concepts of invariant mass, purity and efficiency of a selection, trigger efficiency, and event reconstruction. 
* **Data event reconstruction (Grid computing software)**
  * Turns the pattern of signals from the detector into physics objects, such as jets, photons, and leptons  
  * CPU-intensive task of reducing large quantities of raw data into a form suitable for physics analysis



---
## Setup_Docker_Environment;
Dockerfiles for images that contain ATLAS and ML components with atlas-sit/docker as a dependency at https://gitlab.cern.ch/aml/containers/docker

Install ROOT via Docker and  pull the images from Docker Hub: 
> docker run --rm -it rootproject/root-ubuntu16
> docker pull atlasml/ml-base:debian

To run the ML base container:
> docker run --rm -it -v $PWD:/home/atlas/data -p 8888:8888 atlasml/ml-base:debian

Run a Jupyter server and open jupyter-notebooks running in the container
> jupyter notebook

---

## Performance_Analysis_Plot_Comparison;

Produce plots of the difference between **original and uncompressed variables for each entry of the dataset, divided by the original variable**. You can also add other plots (eg reconstruction loss).

**Performance Analysis**
* Will reduce to disk resources needed to store the ATLAS data 
* Reduces storage space 

**Does the network work well for compression based on them?**
* Compare the results 
* Use graphs for each AE performance 

![dsada](https://github.com/lucylow/CERN_HEP_Autoencoder/blob/master/images/Screen%20Shot%202020-03-11%20at%208.11.48%20PM.png)  


![dsada](https://github.com/lucylow/CERN_HEP_Autoencoder/blob/master/images/Screen%20Shot%202020-03-11%20at%208.12.01%20PM.png)  



![dsada](https://github.com/lucylow/CERN_HEP_Autoencoder/blob/master/images/Screen%20Shot%202020-03-11%20at%208.12.11%20PM.png)  



![dsada](https://github.com/lucylow/CERN_HEP_Autoencoder/blob/master/images/Screen%20Shot%202020-03-11%20at%208.12.20%20PM.png)  


---

## Future_Experimental_Suggestions &#x1F499;

* Discrete variables could be treated more appropriately, e.g. using one-hot
encoding
* Explore other NN architectures like using skipping connections, auxiliary inputs,
multiple branches, more/fewer nodes, more/fewer layers
* Compress event-by-event instead of jet-by-jet. This is commonly used in recurrent neural networks (RNNs)
* Try compressing a similar autoencoder network on gravitational 4d wave data (time series, difficult to
compress)
* ”Chain” this kind of compression to other lossless/lossy compression algorithms being used in ATLAS
* Show that this compression algorithm will not create a significant number of spurious jets or make spurious jets look normal jets (https://github.com/ceres-solver/ceres-solver/commit/d05515b3eb27e2f3880884a878354db035006999)
* Train on whole events and not only on individual jets
* Add more robust scripts for extraction from the raw ROOT data like actual scripts and not jupyter-notebooks for 4 dimensions




---

## References &#x1F499;
* First Ever Open Access Data From the Large Hadron Collider Helped Physicists Confirm Subatomic Patterns https://futurism.com/first-ever-open-access-data-from-the-large-hadron-collider-helped-physicists-confirm-subatomic-patterns
* Unpack the Jet dataset: https://stackoverflow.com/questions/24906126/how-to-unpack-pkl-file
* https://github.com/root-project/root
* A Roadmap for HEP Software and Computing R&D for the 2020s
 https://arxiv.org/abs/1712.06982
* ML compression of ATLAS triggered jet events using autoencoders https://github.com/Skelpdar/HEPAutoencoders
* Eric Wulff. Deep Autoencoders for Compression in High Energy Physics Paper https://lup.lub.lu.se/student-papers/search/publication/9004751 and https://indico.cern.ch/event/870013/contributions/3668675/attachments/1960856/3260728/Autoencoders_for_jet_compresion_in_HEP.pdf 
* FastAI Autoencoders https://alanbertl.com/autoencoder-with-fast-ai/
* Accessing a ROOT TTree With a TBrowser https://root.cern.ch/tutorials
* New Approaches to High Level Particle Physics Analysis https://indico.ph.ed.ac.uk/event/66/contributions/832/attachments/698/853/200215__Analysis_2.0__Python__HEP__...3.pdf
* FAST HEP Helping turn your trees into tables (ie. reads ROOT TTrees, writes summary Pandas DataFrames) https://github.com/fast-hep/fast-carpenter
* Demo analysis based on the CMS Public HEP Tutorial from 2012 https://gitlab.cern.ch/fast-hep/public/fast_cms_public_tutorial
* Docker FAST HEP https://hub.docker.com/r/fasthep/fast-hep-docker
* CMS HEP and it's Four Tutorials with ROOT framework  http://ippog.org/resources/2012/cms-hep-tutorial
* Listing of useful learning resources for machine learning applications in high energy physics (HEPML)
https://github.com/iml-wg/HEP-ML-Resources
