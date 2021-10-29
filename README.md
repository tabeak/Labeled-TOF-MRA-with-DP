# Towards sharing brain images: Differentially private TOF-MRA images with segmentation labels using generative adversarial networks

## Aim

The aim of the project is to generate shareable TOF-MRA image patches and corresponding segmentation labels with differential privacy guarantees. 

## Files

* [config.py](config.py): configuration file for both GAN and U-net
* [run_gan.py](run_gan.py): script for running the GAN with differential privacy
* [generate_samples.py](generate_samples.py): script synthesizing image-label pairs from GAN model
* [model.py](model.py): GAN model file
* [run_unet.py](run_unet.py): script for running one or more U-nets based on ([Livne et al.](https://www.frontiersin.org/articles/10.3389/fnins.2019.00097/full))
* [eval_unet.py](eval_unet.py): script evaluating all specified U-nets on the validation set and the U-net with best validation performance on the test set. All results are printed into a .txt-file.
* [utils](utils): folder containing all helper functions 
* [privacy-gan.yml](privacy-gan.yml): conda environment to be able to run all scripts

## How to run

1. Install the conda environment using the privacy-gan.yml file
2. Specify configurations and paths in config.py
3. To run the GAN with differential privacy, specify the hyperparameters, e.g.:
    ```python
    python3 run_gan.py \
        --trial "GAN1" \               # trial name
        --noisem 1 \                   # noise multiplier value
        --max_norm_dp 1 \              # value for DP noise clipping
        --clip_param_W 0.01 \          # value for WGAN clipping
        --n_discr 5 \                  # number of D updates
        --batch_size 32 \              # batch size
        --lrd 0.00005 --lrg 0.00005 \  # learning rate D and G
        --ndf 96 --ngf 96 \            # number of filters D and G
        --kd 4 --kg 4 \                # kernel size D and G
        --strd 2 --strg 2 \            # strides D and G
        --opg 0 --padg 1 \             # (output) padding G
        --epochs 50 \                  # number of epochs
        --seed 12                      # number for random seed
    ```
4. For generating image-labels from the trained generator, run the generate_samples.py-file, e.g.:
    ```python
    python3 generate_samples.py \
        --trial "GAN1"                 # trial name 
        --epoch 49                     # determine epoch for generation
        --thresh 0.8                   # threshold value for binarizing the label mask
        --lrg 0.00005                  # learning rate G
        --strg 2                       # strides G
        --kg 4                         # kernel size G
        --padg 1                       # padding G
        --opg 0                        # output padding G
    ```
5. The U-net parameters are mostly specify in the config.py-file. Run the U-nets as follows:
    ```python
    python3 run_unet.py \
        --trial "GAN1"                 # trial name
        --epoch 49                     # determine epoch for generation
        --thresh 0.8                   # threshold value for binarizing the label mask
    ``` 
6. Evaluate the U-nets like this:
    ```python
    python3 eval_unet.py \
        --trial "GAN1"                 # trial name
        --epoch 49                     # determine epoch for generation
        --thresh 0.8                   # threshold value for binarizing the label mask
    ``` 

## How to cite

*As soon as the paper is published, this will be updated*
