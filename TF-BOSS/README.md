#	Instructions to run BOSS methodology using Tensorflow

#	Setup Fixmatch
	sudo apt install python3-dev python3-virtualenv python3-tk imagemagick
	virtualenv -p python3 --system-site-packages env3
	. env3/bin/activate
	pip install -r requirements.txt

#	Install datasets
 	setup.sh #currently set for cirfar10, uncomment necessary lines for other datasets	
		 #for more instructions, see fixmatch github repo: https://github.com/google-research/fixmatch

#	designate prototypes
 	svhnData.sh #and/or
	cifarData.sh
	
#	run BOSS
	run.sh #dataset, seed, balance method and hyperparameters can be set in-script	
