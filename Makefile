default: unet 

unet:
	python3 src/unet.py

remove_alpha:
	python3 remove_alpha.py

diffusion_utils:
	python3 src/diffusion_utils.py