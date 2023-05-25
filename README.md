# Super-Resolution-Image
----------------------------------------------------------------------------------------------------------
Dependencies:
	Python 3
	PyTorch >= 1.0 (CUDA version >= 7.5 if installing with CUDA)
	Python packages: pip install numpy opencv-python

----------------------------------------------------------------------------------------------------------
For Testing Model:
----------------------------------------------------------------------------------------------------------
1) Using PyCharm:
	1. Copy 'LR' folder in your current PyCharm project. 
	Place your own low-resolution images in ./LR folder which you want to test. 
	(There are two sample images - baboon and comic).
	2. Copy 'figures' folder in your current PyCharm project.
	3. Copy 'models' folder in your current PyCharm project folder which contains 
	'RRDB_ESRGAN_x4.pth' model.
	4. Copy RRDBNet_arch.py, net_interp.py, transer_RRDB_models.py and test.py files 
	in your current PyCharm project folder.
	5. Run test.py file.
	6. You can see output images in 'results' folder.

----------------------------------------------------------------------------------------------------------
2) Using Google Colab: 
	1. from google.colab import drive
	   drive.mount('/content/gdrive/')
	2. % cd gdrive/My Drive/
	3. ! git clone https://github.com/xinntao/ESRGAN
	4. cd ESRGAN/
	5. Copy 'RRDB_ESRGAN_x4.pth' model in 'ESRGAN/models/' folder.
	6. ! python test.py - Run this command in colab.
	7. You can see output images in 'results' folder.
----------------------------------------------------------------------------------------------------------
