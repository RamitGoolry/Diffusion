default: train_unet_diffusion

train_unet_diffusion:
	python3 src/train_unet_diffusion.py

train_vqgan:
	python3 src/train_vqgan.py

train_unet:
	python3 src/train_unet.py

unet:
	python3 src/unet.py

remove_alpha:
	python3 remove_alpha.py

diffusion_utils:
	python3 src/diffusion_utils.py

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -f test.png
	rm -rf wandb
	echo "Workspace Cleaned"
