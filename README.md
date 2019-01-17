Voice Activity Detection (VAD) is a preprocessing step for several speech processing applications. Handling this task is not a straightforward task in noisy environments since the statistics of the noise are unknown beforehand.

In the presence of stationary noise alone, the spectrum of the signal in short time intervals could be used to measure the spectral
flatness in order to decide on the voice activity. 

However, this approach cannot help when the noise spectrum changes quickly, like in the case of transient noises which are abrupt interferences, such as keyboard taps, knocks.

Speech and Transients may often appear similar when represented by MFCCs in the Euclidean space. The generation of many signals can be associated with small set of physical constraints controlling their production. For example, the generation of speech is controlled by the position of vocal tract and the movement of lips, tongue and jaw. 

It is shown by [Dov et al.](https://israelcohen.com/wp-content/uploads/2018/05/TASLP_Dec2016.pdf) that Mahalanobis distance between the observable signals approximates the Euclidean distance between the underlying generating variables. We use this metric to compute the similarity matrix for *Spectral clustering* that divides given frames into two clusters.(i.e., speech presence and speech absence frames). The eigen vectors of the normalized Laplacian of the similarity matrix and the GMM are utilized to compute the likelihood ratio for voice activity detection. Implementation details can be found in our [report.](https://github.com/varshapendyala/Voice-Activity-Detection/blob/master/voice-activity-detection.pdf)
