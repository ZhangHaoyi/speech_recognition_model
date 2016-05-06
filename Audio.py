#python modules
import matplotlib.pyplot as plt
import librosa
import numpy as np
import sklearn

class audio:

	'''
	This is a class function which essentially is a wrapper for the librosa and sklearn library.
	The Class Audio is defined by:
		_x= loads the time series audio waveform as an array
		_sr= loads the sampling rating

	'''

	#constructors
	def __init__(self):
		self._x=0
		self._sr=0

	'''	
	def __init__(self, _x, _sr):
		self._x=_x
		self._sr=_sr
	'''

	#uses the librosa library to render the audio file
	def load_audio_file (self,file_name):
		try:
			print "Loading File...."
			self._x, self._sr = librosa.load(file_name)
		except:
			print "Error, Unable to load file! Check the file format compatibility"


	#get functions for the files class audio variables
	def getSamplingRate(self):
		samplingRate=self._sr
		return samplingRate

	def getAudioWaveForm(self):
		x=self._x
		return x

	'''
	n_mfcc is an int value >0 and returns the an np array with mel cepstrum 
	coefficent dimensions as the row value
	
	'''
	def convertMFCC(self, n_mfcc):
		try:
			print "Converting Audio Input to MFCC Coefficent Matrix..."
			mfcc_mat=librosa.feature.mfcc(self._x, self_sr)
            print mfcc_mat
			return mfcc_mat
		except:
			print "Error trying to perform the task."

	'''
	Mel Cepstrum coefficient Normalization and scaling
	returns a numpy array with the scaled coefficients
	'''
	def scalingMFCC(self, mfcc_mat):
		try:
			print "Scaling MFCC Values..."
			print mfcc.shape
			scaled_mfcc=sklearn.preprocessing.scale(mfcc_mat, axis=1)
			print scaled_mfcc
			return scaled_mfcc
		except:
			print "Unable to perform scale. Check the Dimensions"


	'''
	Plotting functions:
	these functions are responsible for handling the graphs
	'''

	
	#Plotting the WavePlot as as simple wave spectrogram
	def printWavePlot(self):
		librosa.display.waveplot(self._x, sr=self._sr)


	#Plotting the MFCC Spectrogram 
	def plotMFCC(self,mfcc_mat):
		librosa.display.specshow(mfcc_mat,sr=self._sr, x_axis='time')

