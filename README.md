<h2>Tensorflow-Image-Segmentation-Pre-Augmented-Mammogram-Density (2025/05/09)</h2>

Sarah T. Arai<br>
Software Laboratory antillia.com<br><br>

This is the first experiment of Image Segmentation for Mammogram-Density
 based on 
the latest <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, 
and a pre-augmented <a href="https://drive.google.com/file/d/1WBjG65EXgTf-r0Ky4EhYhIXe2IbbyspP/view?usp=sharing">
Mammogram-Density-ImageMask-Dataset.zip</a>, which was derived by us from  
<a href="https://data.mendeley.com/datasets/tdx3h2fn9v/4#:~:text=This%20dataset%20consists%20of%20mammogram%20images%2C%20complete%20with,and%20breast%20area%20annotated%20by%20an%20expert%20radiologist.
">
Mammogram Density Assessment Dataset
</a>
<br>
<br>

<b>Data Augmentation Strategy:</b><br>
 To address the limited size of Mammogram Density Assessment Dataset, which contains 596 images and their corresponding dense_masks in the train dataset, 
 we employed <a href="./generator/ImageMaskDatasetGenerator.py">an offline augmentation tool</a> to generate a 512x512 pixels pre-augmented dataset, which supports the following augmentation methods.
<br>
<li>Vertical flip</li>
<li>Horizontal flip</li>
<li>Shrinks</li>
<li>Shears</li> 
<li>Deformation</li>
<li>Distortion</li>
<li>Barrel distortion</li>
<li>Pincushion distortion</li>
<br>
Please see also the following tools <br>
<li><a href="https://github.com/sarah-antillia/Image-Deformation-Tool">Image-Deformation-Tool</a></li>
<li><a href="https://github.com/sarah-antillia/Image-Distortion-Tool">Image-Distortion-Tool</a></li>
<li><a href="https://github.com/sarah-antillia/Barrel-Image-Distortion-Tool">Barrel-Image-Distortion-Tool</a></li>
<br>
<hr>
<b>Actual Image Segmentation for Images of 512x512 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks. <br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/mini_test/images/178.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/mini_test/masks/178.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/mini_test_output/178.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/mini_test/images/286.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/mini_test/masks/286.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/mini_test_output/286.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/mini_test/images/346.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/mini_test/masks/346.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/mini_test_output/346.jpg" width="320" height="auto"></td>
</tr>
</table>

<hr>
<br>
In this experiment, we used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this Mammogram-Density Segmentation Model.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>

<br>

<h3>1. Dataset Citation</h3>

The dataset used here has been take from the following web site:<br> 
<a href="https://data.mendeley.com/datasets/tdx3h2fn9v/4#:~:text=This%20dataset%20consists%20of%20mammogram%20images%2C%20complete%20with,and%20breast%20area%20annotated%20by%20an%20expert%20radiologist.">
Mammogram Density Assessment Dataset
</a><br>
<br>
<b>Published:</b> 8 April 2024 Version 4<br>

<b>DOI:</b>10.17632/tdx3h2fn9v.4<br>
<br>
<b>Contributors:</b><br>

Hamid Behravan, Naga Raju Gudhe, Hidemi Okuma, Arto Mannermaa<br>
<br>
<b>Description</b><br>
This dataset consists of mammogram images, complete with corresponding segmentation masks for dense tissue <br>
and breast area annotated by an expert radiologist. <br>
<br>
<!--
<b>Files</b><br>

<b>train.zip:</b><br>
 Comprises three sub-folders: 'images', 'breast_masks', and 'dense_masks'. <br>
The 'images' sub-folder houses the original images. The 'breast_masks' and 'dense_masks' sub-folders contain <br>
the ground truth segmentation masks for the breast area and dense tissue segmentation, respectively. <br>
All images are in JPG format. All masks and the corresponding images have the same dimension.<br>
<br>
<b>test.zip:</b><br>
Contains the images for test set in JPG format. No ground truths are provided for the test set.<br>
<br>
<b>train.csv:</b><br> 
The training set filelist consists of two columns. The first column is the ‘Filename’, and the second column is the ‘Density',<br>
 the ground truth for the breast density prediction task.<br>
<br>
<b>test.csv:</b><br>
The test set filelist contains the filenames of the test sets.<br>
<br>
-->
This dataset can be utilized for tasks such as segmentation and breast density estimation. <br>
The mammograms were sourced from the public VinDr-Mammo dataset, which can be found at [this link]<br>
(https://vindr.ai/datasets/mammo). We have given annotations, including both segmentation masks and <br>
density values, for this public dataset.<br>
<br>
If you use this dataset in your research or other purposes, please cite the following studies:<br><br>

Gudhe, N.R., Behravan, H., Sudah, M. et al.<br>
<b> Area-based breast percentage density estimation in mammograms
 using weight-adaptive multitask learning.</b><br> 
 Sci Rep 12, 12060 (2022). <br>
 https://doi.org/10.1038/s41598-022-16141-2<br>
<br>
Hieu T. Nguyen et al. <br>
<b>A large-scale benchmark dataset for computer-aided diagnosis in full-field digital mammography.</b> <br>
2022. https://doi.org/10.1101/2022.03.07.22272009<br>
<br>
Related Links<br>
Article<br>
https://www.nature.com/articles/s41598-022-16141-2<br>
is related to this dataset<br>
<br>
<b>Licence</b> CC BY 4.0<br>


<br>
<h3>
<a id="2">
2 Mammogram-Density ImageMask Dataset
</a>
</h3>
 If you would like to train this Mammogram-Density Segmentation model by yourself,
 please download the dataset from the google drive  
<a href="https://drive.google.com/file/d/1WBjG65EXgTf-r0Ky4EhYhIXe2IbbyspP/view?usp=sharing">
Mammogram-Density-ImageMask-Dataset.zip</a>
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─Mammogram-Density
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>

On the derivation of this dataset, please refer to the following Python scripts:
<li><a href="./generator/ImageMaskDatasetGenerator.py">ImageMaskDatasetGenerator.py</a></li>
<li><a href="./generator/split_master.py">split_master.py</a></li>
<br>
The folder structure of the original Mammogram-Density is the following.<br>
<pre>
./Mammogram Density Assessment Dataset
├─test
│  └─images
└─train
    ├─breast_masks
    ├─dense_masks
    └─images
</pre>
We derived our 512x512 pixels ImageMask Dataset from dense_masks and images of 2Kx3K pixels in train dataset by using <a href="./generator/ImageMaskDatasetGenerator.py">
ImageMaskDatasetGenerator.py</a>. 
<br>
<br>
<b>Mammogram-Density Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/Mammogram-Density_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is enough to use for a training set of our segmentation model.
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorflowUNet Model
</h3>
 We have trained Mammogram-DensityTensorflowUNet Model by using the following
<a href="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16 </b> and large <b>base_kernels = (9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dropout_rate   = 0.05
dilation       = (1,1)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.0001
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation tool. 
<pre>
[model]
model         = "TensorflowUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "bce_dice_loss" and "dice_coef".<br>
<pre>
[model]
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
</pre>
<b >Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/asset/epoch_change_infer_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (epoch 86,87,88)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/asset/epoch_change_infer_end.png" width="1024" height="auto"><br>
<br>

In this experiment, the training process was stopped at epoch 88  by EarlyStopping Callback.<br><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/asset/train_console_output_at_epoch_88.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for Mammogram-Density.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/asset/evaluate_console_output_at_epoch_57.png" width="720" height="auto">
<br><br>Image-Segmentation-Mammogram-Density

<a href="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/evaluation.csv">evaluation.csv</a><br>

The loss (bce_dice_loss) to this Mammogram-Density/test was not low, and dice_coef not high as shown below.
<br>
<pre>
loss,0.1194
dice_coef,0.808
</pre>
<br>

<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for Mammogram-Density.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks </b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/mini_test/images/12.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/mini_test/masks/12.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/mini_test_output/12.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/mini_test/images/96.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/mini_test/masks/96.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/mini_test_output/96.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/mini_test/images/113.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/mini_test/masks/113.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/mini_test_output/113.jpg" width="320" height="auto"></td>
</tr>


<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/mini_test/images/178.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/mini_test/masks/178.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/mini_test_output/178.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/mini_test/images/346.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/mini_test/masks/346.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/mini_test_output/346.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/mini_test/images/412.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/mini_test/masks/412.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram-Density/mini_test_output/412.jpg" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>


<h3>
References
</h3>
<b>1. Fully Automated Breast Density Segmentation and Classification Using Deep Learning</b><br>
Nasibeh Saffari, Hatem A Rashwan, Mohamed Abdel-Nasser, Vivek Kumar Singh, Meritxell Arenas,<br>
 Eleni Mangina, Blas Herrera, Domenec Puig<br>
<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC7700286/">
https://pmc.ncbi.nlm.nih.gov/articles/PMC7700286/
</a>
<br>
<br>
<b>2. Breast Dense Tissue Segmentation with Noisy Labels: A Hybrid Threshold-Based and Mask-Based Approach</b><br>
Andrés Larroza, Francisco Javier Pérez-Benito, Juan-Carlos Perez-Cortes, Marta Román, Marina Pollán,<br>
Beatriz Pérez-Gómez, Dolores Salas-Trejo, María Casals and Rafael Llobet<br>
<a href="https://www.mdpi.com/2075-4418/12/8/1822">
https://www.mdpi.com/2075-4418/12/8/1822
</a>
<br>
<br>

