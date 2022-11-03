Prediction of Photometric Redshift from galaxy images using SDSS Dataset

-------------------------------------------------------------------------------------------------------------------

The Dataset is given in Data folder.
The logs and model parameters with metric plots are in the folder logs.

There are 6 python files in this project. 

1. image_generator.py : script to visualise the datapoints from the dataset and to create 				reconstruction image from ugriz filters. Depends on the file img_scale.py
			for image procressing directives. Usage :
			
			$ python3 image_generator.py
			Enter the sample to generate :  (sample # from dataset)
---------------------------------------------------------------------------------------------------
	
2. img_scale.py : utility file for astronomical image preprocessing. Borrowed from (https://astromsshin.github.io/science/code/Python_fits_image/index.html). No direct usage.
---------------------------------------------------------------------------------------------------
3. trainer.py : main script to train and test models with dataset without using mixed input.
		Loads dataset and divides into test set and trainset.Uses pytorch as backend for 			creating neural networks. Prerequisites - models.py, utils.py
		Usage :
		
		$ python3 trainer.py
---------------------------------------------------------------------------------------------------
4. mm_trainer.py : modified version of trainer.py to handle the models that take extra mixed input. 			   Works exactly like trainer for model training, testing and metrics evaluation. 
		   Prerequisites - models.py, utils.py
		   Usage :
		   	
		   $ python3 mm_trainer.py
----------------------------------------------------------------------------------------------------
5. models.py : Utility file to store different Deep CNN models. Requires pytorch for model 		       definition. No direct usage. 
----------------------------------------------------------------------------------------------------
6. utils.py : Utility file to store function definitions for various tasks like - dataset creation,  		      metrics computations, Plot generation, etc. Prerequisites - numpy, pytorch, matplotlib.
 	      Usage -  No direct usage for other files although running standalone will produce output
 	      that will verify the integrity of function(works as unit test).
 	      
 	      $ python3 utils.py
------------------------------------------------------------------------------------------------------
		
