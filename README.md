# Deep Neural Network Autoencoders for Data Compression in High Energy Physics 🔬&#x1F499;

**＊ ✿ ❀ Use an autoencoder to compress data from CERN's high energy physics dataset ❀ ✿ ＊**


<div>
  
  [![Status](https://img.shields.io/badge/status-active-success.svg)]()
  [![GitHub Issues](https://img.shields.io/github/issues/lucylow/CERN_HEP_Autoencoder.svg)](https://github.com/lucylow/CERN_HEP_Autoencoder/issues)
  [![GitHub Pull Requests](https://img.shields.io/github/issues-pr/lucylow/CERN_HEP_Autoencoder.svg)](https://github.com/lucylow/CERN_HEP_Autoencoder/pulls)
  [![License](https://img.shields.io/aur/license/android-studio.svg)]()

</div>


---

## Table_of_Contents &#x1F499;

* [Motivation](#Motivation-)
* [Autoencoders](#Autoencoders-)
* [Download the HEP data](#Download_the_HEP_data-)
* [Run the training script](#Run_the_training_script-) 
* [Loss error function](#Loss_error_function-)
* [TensorBoard monitoring model training](#TensorBoard_monitoring_model_training-)
* [Conclusion Model Discussion](#Conclusion_Model_Discussion-)
* [References](#references-) 

---

## Motivation &#x1F499;

* Use an autoencoder (AE) to compress hadron jet event data from 4 to 3 variables. https://drive.google.com/drive/folders/1JaCB-prsDhEX4Ovk-UjC9bMxOHbpfREr?usp=sharing
* Introduction to fundamental concepts of data analysis in HEP experiments given a basic knowledge of particle physics
* Technical Requirements:
  * PyTorch
  * FastAI Library https://docs.fast.ai/install.html
  * ROOT Data Analysis Framework 
  * ML environment with CERN ATLAS and docker containters

---
## High Energy Physics  &#x1F499;
* In  physical sciences, physicists study subatomic particles to gain an understanding of how particles smaller than an atom interact by analyzing collision data
* Standard Model of elementary particles 
  * 6 "flavors" of quarks: up, down, strange, charm, bottom, and top;
  * 6 types of leptons: electron, electron neutrino, muon, muon neutrino, tau, tau neutrino;
  * 12 gauge bosons: the photon of electromagnetism, the three W and Z bosons of the weak force, and the eight gluons of the strong force;
  * The Higgs boson
* Sub-atomic particles 
  * Bosons 
  * Hadrons
  * Fermions
* Hadrons 
  * Subatomic composite particle made of two or more quarks held together by the strong force in a similar way as molecules are held together by the electromagnetic force
  * Two types of Hardons:
    * Baryons Ex. Protons and neutrons 
    * Mesons Ex. Pions
  * Contain few (≤ 5) antiquarks
  * Like all subatomic particles, hadrons are assigned quantum numbers corresponding to the representations of the Poincaré group: JPC(m), where J is the spin quantum number, P the intrinsic parity (or P-parity), C the charge conjugation (or C-parity), and m the particle's mass.
  * All composite particles contain multiple quarks (antiquarks) bound together by gluons 
  * Unstable and eventually decay (break down) into other particles
  * Experimentally, hadron physics is studied by colliding protons or nuclei of heavy elements such as lead or gold, and detecting the debris in the produced particle showers. 
  * In the natural environment, mesons such as pions are produced by the collisions of cosmic rays with the atmosphere

---
## CERN's Large Hadron Collider &#x1F499;
* Large Hadron Collider (LHC) = world's largest and highest-energy particle collider built by European Organization for Nuclear Research (CERN) in Geneva 
*  British scientist Tim Berners-Lee invented the World Wide Web (WWW) in 1989, while working at CERN originally to meet the demand for automated information-sharing between scientists around the world
* The aim of the LHC's detectors is to allow physicists to test the predictions of different theories of particle physics, including measuring the properties of the Higgs boson. The collider is a type of a particle accelerator with two directed beams of particles. The research tool used to accelerate particles to very high kinetic energies and let them impact other particles
* Goal is to understand more of the subatomic world and the laws of physics that govern them and answer some of the fundamental open questions in science. This requires high energy collisions and particle decays after very short periods of time.

----

## ATLAS &#x1F499;
  * Largest general-purpose particle detector experiment at the Large Hadron Collider
  * Designed to measure the broadest possible range of signals. This is intended to ensure that whatever form any new physical processes or particles might take, ATLAS will be able to detect them and measure their properties. ATLAS is designed to detect these particles, namely their masses, momentum, energies, lifetime, charges, and nuclear spins and to search for evidence of theories of particle physics beyond the Standard Model.
  * One of two general-purpose detectors. ATLAS studies the Higgs boson and looks for signs of new physics, including the origins of mass and extra dimensions.
  * Involved in the discovery of the Higgs boson in July 2012. One of the most important goals of ATLAS was to investigate a missing piece of the Standard Model, the Higgs boson
  * Higgs mechanism is essential to explain the generation mechanism of the property "mass" for gauge bosons.
  * Peter Higgs and François Englert had been awarded the 2013 Nobel Prize in Physics after serach found Higgs boson
  
---
## ATLAS Data Generation &#x1F499;
* Generates large amounts of data ~ total of 1 petabyte of raw data per second. 25 megabytes per event (raw; zero suppression reduces this to 1.6 MB), multiplied by 40 million beam crossings per second in the center of the detector
* Particle Physics Trigger System 
  * Due to the extremely high LHC collision rate of up to 20 MHz not all events can be stored. A trigger
system selects the ”interesting” events and reduces the total event rate to a few hundred Hertz.
  * When the LHC is operating, 40 million packets of protons collide every second at the centre of the ATLAS detector. Every time there is a collision, the ATLAS Trigger selects interesting collisions and writes them to disk for further analysis. A small subset of these collisions are passed through visualisation software and displayed on a large screen in the ATLAS Control Room for the physicists on shift to view. Although this is just another tool for monitoring the detector, it is also fun to watch. So, we thought we would share it with you here. (Be patient, it might take a moment to load.)
  * System decides which events in a particle detector to keep when only a small fraction of the total can be recorded
  * Selectivity of the trigger: The ratio of the trigger rate to the event rate   * For example, the Large Hadron Collider (LHC) has an event rate of 40 MHz (4·107 Hz), and the Higgs boson is expected to be produced there at a rate of roughly 1 Hz. 
  * Two trigger levels 
    * Level 1 :  uses information from the calorimeters and the muon spectrometer, and decreases the rate of events in the read-out to 100 kHz
    * Level 2 : uses limited regions of the detector, so-called regions of interest (RoI), to reconstruct events by matching energy deposits to tracks
* The remaining data, corresponding to about 1000 events per second, are stored for further analysis


---

## Autoencoders &#x1F499;
* "Autoencoding" == **Data compression algorithm** with compression and decompression functions
* Self-supervised learning where target models are generated from input data
* Implemented with **neural networks** - useful for problems in unsupervised learning (no labels)



---

## Technical Download_the_HEP_data &#x1F499;
* Download dataset https://drive.google.com/drive/folders/1JaCB-prsDhEX4Ovk-UjC9bMxOHbpfREr
* Uncompress the large file size (26 MBytes)
* Move files into `dataset` folder in the example folder of this repo
* Data is in a pickle python format: 

> import pandas
> object = pd.read_pickle(r'filepath')

---
## Data Analysis &#x1F499;

The event information is kept to a minimum: four-momentum components of leading jets, electrons, muons, and photons; combined particle based isolation for leptons and photons as well as b-tag information.

We've chosen as an example a TTbar analysis to explain the concepts of invariant mass, purity and efficiency of a selection, trigger efficiency, and event reconstruction. The goal is a simple cross section measurement and a top quark mass measurement.

The starting point is an introduction to the analysis framework, including examples for producing histograms of basic quantities such as momentum distributions. The students are then supposed to develop the analysis by themselves, following some guidelines and suggestions provided through exercise sheets. A sample solution can be provided as well (which can be handed to the students at the very end).

The full analysis, including data and MC files fits into a 30 Megabyte tar ball (see below) and runs on a standard computer within a few seconds. The only requirement for the computing environment is a ROOT installation.

* Data event reconstruction
  * Performed on all permanently stored events, turning the pattern of signals from the detector into physics objects, such as jets, photons, and leptons.  
  * Grid computing software is being used extensively for event reconstruction, allowing the parallel use of university and laboratory computer networks throughout the world for the CPU-intensive task of reducing large quantities of raw data into a form suitable for physics analysis. 


* Data Event Sonification 
  * One can also take the physical parameters of collision data and transform them to sound, a process called sonification. The Quantizer project, developed in collaboration with MIT Media Lab, applies pre-defined mappings, developed by musical composers, to create unique real-time streams of, well, the songs of nature. 


---
## Setup Docker Environment  &#x1F499;
Dockerfiles for images that contain ATLAS and ML components with atlas-sit/docker as a dependency at https://gitlab.cern.ch/aml/containers/docker

Also need to install ROOT via Docker 
> docker run --rm -it rootproject/root-ubuntu16

Download DOCKER and pull the images from Docker Hub:
> docker pull atlasml/ml-base:debian

To run the ML base container:
> docker run --rm -it -v $PWD:/home/atlas/data -p 8888:8888 atlasml/ml-base:debian

Run a Jupyter server and open jupyter-notebooks running in the container
> jupyter notebook


---
## Installation &#x1F499;
 
Git pull the project from the git repository:

git init
git pull https://github.com/lucylow/CERN_HEP_Autoencoder

--- 

## Data preprocessing &#x1F499;

Pre-processing: Extract data from the /datasets

The data comes in two types: 4-dim data and the 27-dim data. (Although the original events holds 29 values, only 27 of them are of constant size.)

The raw DxAODs can be processed into a 4-dimensional dataset with process_ROOT_4D.ipynb, where the data is pickled into a 4D pandas Dataframe. process_ROOT_27D.ipynb does the same for the 27-dimensional data. Since pickled python objects are very version incompatible, it is recommended to process the raw ROOT DxAODs instead of providing the pickled processed data.

For ease of use, put raw data in data/ and put processed data in processed_data/

The 27-variables in question are:

All processed data will be placed in processed_data/ after extraction (by default). No normalization or other ML-related pre-processing is done in this step.

Training: An (uncommented) example of training a 4D-network is fastai_AE_3D_200_no1cycle.ipynb and looks very much like every other training script in this project. If the data you have looks any different it will need to be retrained. See the report for previous searches of optimal network sizes.

Analysis and plots: An example of running a 4-dimensional already trained network is 4D/fastai_AE_3D_200_no1cycle_analysis.ipynb For an example of analysing a 27-D network is 27D/27D_analysis.py.

Code structure: The folders named 4D/, 25D/ and 27D/ simply holds training analysis scripts for that amount of dimensions.

nn_utils.py holds various heplful for networks structures and training functions.

utils.py holds functions for normalization and event filtering, amongst others.

---

## Run_the_training_script &#x1F499;

* For this project, we don't need to train the network since there is an ready-made network in the folder. However, there is a training/testing dataset if you want to train your own model:

* Can not feed all the data to the model at once due to computer memory limitations so **data is split into "batches"** 
* When all batches are fed exactly once, an "epoch" is completed. As training script runs, **preview images afer every epoch will show**

---

## Performance Analysis Plot Comparison  &#x1F499;

Performance Analysis
-	Will reduce to disk resources needed to store the TALAS data 
-	Reduces storage space 



Produce plots of the difference between original and uncompressed variables for each entry of the dataset, divided by the original variable. You can also add other plots (eg reconstruction loss).


does the network work well for compression based on them?

-	Compare the results 
-	Use graphs for each AE performance 


---

## Conclusion_Model_Discussion &#x1F499;

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


Analysis scripts for CPU/GPU and memory usage when evaluating the networks?

Adding more robust scripts for extraction from the raw ROOT data, i.e. actual scripts and not jupyter-notebooks, for 4, 25 and 27 dimensions. (And optimize them.)?

Chain networks with other compression techniques?

Train on whole events and not only on individual jets?



Discovery of the Higgs boson has opened up whole new windows in the search for new physics, since its properties are predicted to be different in different theoretical models. Supersymmetry, for example, predicts the existence of at least five different types of Higgs bosons. Will the Standard Model continue to survive the precision measurements of the LHC or will an improved model appear? Only the analysis of new data at even higher collision energy will tell.



Physicists belonging to worldwide collaborations work continuously to improve detector-calibration methods, and to refine processing algorithms to detect ever more interesting events.

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
