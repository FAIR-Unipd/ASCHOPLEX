import os
import nibabel as nb
import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy
from monai.apps.utils import get_logger
from typing import Any, Dict, Optional
from monai.config import print_config
from monai.bundle.config_parser import ConfigParser
from monai.data import CacheDataset, DataLoader, load_decathlon_datalist, decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNETR, DynUNet
from monai.transforms import (
    Activationsd,
    EnsureChannelFirstd,
    AsDiscreted,
    Compose,
    LoadImaged,
    Spacingd,
    Orientationd,
    CastToTyped,
    NormalizeIntensityd,
    Invertd,
    CopyItemsd,
)
from monai.utils.enums import AlgoEnsembleKeys
from monai.utils import set_determinism
from monai.transforms import SaveImage
from monai.auto3dseg.utils import concat_val_to_np
from monai.apps.auto3dseg.utils import import_bundle_algo_history
from monai.apps.auto3dseg import (
    AlgoEnsembleBuilder,
    AlgoEnsemble,
)


print_config()

set_determinism(seed=123)
logger = get_logger(module_name=__name__)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

join = os.path.join

def set_prediction_params(params: Optional[Dict[str, Any]] = None):
        """
        Set the prediction params for all algos.

        Args:
            params: a dict that defines the overriding key-value pairs during prediction. The overriding method
                is defined by the algo class.

        Examples:

            For BundleAlgo objects, this set of param will specify the algo ensemble to only inference the first
                two files in the testing datalist {"file_slices": slice(0, 2)}

        """
        if params is None:
            pred_params = {"sigmoid": True}  # output will be 0-1
        else:
            pred_params = deepcopy(params)

def set_image_save_transform(output_dir, kwargs):
        """
        Set the ensemble output transform.

        Args:
            kwargs: image writing parameters for the ensemble inference. The kwargs format follows SaveImage
                transform. For more information, check https://docs.monai.io/en/stable/transforms.html#saveimage .

        """

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Directory {output_dir} is created to save ensemble predictions")

        if "output_postfix" in kwargs:
            output_postfix = kwargs.pop("output_postfix")
        else:
            output_postfix = "ensemble_seg"

        return SaveImage(output_dir=output_dir, output_postfix=output_postfix, separate_folder=False)


class Predict_Directly:

    # Directly predict the Choroid Plexus segmentations without running a finetuning step
    
    # initialization
    def __init__(self, work_dir: str = ".", json_file: str = ".", output_dir=None):

        self.Workdir=work_dir
        self.JSON_file=json_file
        if output_dir is None:
            self.Output_dir=join(self.Workdir, 'working_directory_prediction')
        elif  isinstance(output_dir, str):
            self.Output_dir=output_dir
        self.Prediction_path=[]
       
    def predict(self):

        work_dir = self.Output_dir

        # create working directory
        if not os.path.isdir(work_dir):
            os.makedirs(work_dir)

        algorithm_path=join(self.Workdir, 'DNN_models/algorithm_trained')

        # Testing transforms
        testing_transforms = Compose(
        [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear"), align_corners=True),
                Orientationd(keys=["image"], axcodes="RAS"),
                CastToTyped(keys=["image"], dtype= np.float32),
                NormalizeIntensityd(keys=["image"], channel_wise=True, nonzero=True),
                CastToTyped(keys=["image"], dtype= np.float32),
        ]
        )
    
        # Load dataset
        # JSON file --> contains dataset directory
        datasets =  self.JSON_file
        test_files = load_decathlon_datalist(datasets, True, "testing")
        test_ds = CacheDataset(
            data=test_files, transform=testing_transforms, cache_num=6, cache_rate=1.0, num_workers=4
        )
        test_loader = DataLoader(
            test_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
        )

        # load trained models inside model configurations
        # Network architectures (model_0-->model_4)
        model_UNETR_0=UNETR(
        dropout_rate= 0.0, 
        feature_size= 16, 
        hidden_size= 768,
        img_size= (128,128,128), 
        in_channels= 1, 
        mlp_dim= 3072, 
        norm_name= "instance",
        num_heads= 12, 
        out_channels= 2, 
        pos_embed= "perceptron", 
        res_block= True,
        ).to(device)

        model_UNETR_3=UNETR(
        dropout_rate= 0.0, 
        feature_size= 16, 
        hidden_size= 768,
        img_size= (128,128,128), 
        in_channels= 1, 
        mlp_dim= 3072, 
        norm_name= "instance",
        num_heads= 12, 
        out_channels= 2, 
        pos_embed= "perceptron", 
        res_block= True,
        ).to(device)

        model_DynUNet_1=DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            kernel_size=(3, (1, 1, 3), 3, 3),
            strides=(1, 2, 2, 1),
            upsample_kernel_size=(2, 2, 1),
            norm_name=("INSTANCE", {"affine": True}),
            deep_supervision=False,
            deep_supr_num=1,
            res_block=False,
        ).to(device)
        
        model_DynUNet_2=DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            kernel_size=(3, (1, 1, 3), 3, 3),
            strides=(1, 2, 2, 1),
            upsample_kernel_size=(2, 2, 1),
            norm_name=("INSTANCE", {"affine": True}),
            deep_supervision=False,
            deep_supr_num=1,
            res_block=False,
        ).to(device)

        model_DynUNet_4=DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            kernel_size=(3, (1, 1, 3), 3, 3),
            strides=(1, 2, 2, 1),
            upsample_kernel_size=(2, 2, 1),
            norm_name=("INSTANCE", {"affine": True}),
            deep_supervision=False,
            deep_supr_num=1,
            res_block=False,
        ).to(device)
    
        fold_0 = join(algorithm_path, "UNETR_128_DiceCE_0", "model_fold0", "best_metric_model.pt")
        fold_1 = join(algorithm_path, "DynUnet_128_DiceCE_1", "model_fold1", "best_metric_model.pt")
        fold_2 = join(algorithm_path, "DynUnet_128_Dice_2", "model_fold2", "best_metric_model.pt")
        fold_3 = join(algorithm_path, "UNETR_128_DiceCE_3", "model_fold3", "best_metric_model.pt")
        fold_4 = join(algorithm_path, "DynUnet_128_DiceCE_4", "model_fold4", "best_metric_model.pt")


        models=[]

        model_f0=torch.load(fold_0)
        model_UNETR_0.load_state_dict(model_f0)
        model_UNETR_0.eval()
        models.append(model_UNETR_0)
        
        model_f1=torch.load(fold_1)
        model_DynUNet_1.load_state_dict(model_f1)
        model_DynUNet_1.eval()
        models.append(model_DynUNet_1)
        
        model_f2=torch.load(fold_2)
        model_DynUNet_2.load_state_dict(model_f2)
        model_DynUNet_2.eval()
        models.append(model_DynUNet_2)
        
        model_f3=torch.load(fold_3)
        model_UNETR_3.load_state_dict(model_f3)
        model_UNETR_3.eval()
        models.append(model_UNETR_3)
        
        model_f4=torch.load(fold_4)
        model_DynUNet_4.load_state_dict(model_f4)
        model_DynUNet_4.eval()
        models.append(model_DynUNet_4)

        # ensemble predictions
        post_pred=Compose(
            [
                Invertd(
                    keys="pred",
                    transform=testing_transforms,
                    orig_keys="image",
                    meta_keys="pred_meta_dict",
                    orig_meta_keys="image_meta_dict",
                    meta_key_postfix="meta_dict",
                    nearest_interp=False,
                    to_tensor=True,
                ),
                Activationsd(keys="pred", softmax=True, sigmoid=False),
                CopyItemsd(keys="pred", times=1, names="pred_final"),
                AsDiscreted(keys="pred_final", argmax=True),
            ]
            ) 
        name_file='predicted_seg.nii.gz'
        name_folder=["UNETR_0", "DynUnet_1", "DynUnet_2", "UNETR_3", "DynUnet_4"]
        epoch_iterator_val = tqdm(test_loader)

        with torch.no_grad():
            for step, batch in enumerate(epoch_iterator_val):

                name_image = test_ds[step]["image_meta_dict"]["filename_or_obj"]
                # patient_id=os.path.basename(os.path.dirname(name_image))
                if name_image.endswith('.nii'):
                    patient_id=os.path.basename(name_image).rstrip('image.nii')
                elif name_image.endswith('.nii.gz'):
                    patient_id=os.path.basename(name_image).rstrip('image.nii.gz')
                else:
                    raise ValueError("Data are not in the correct format. Please, provide images in .nii or .nii.gz Nifti format")

                img=nb.load(name_image)
                val_outputs_all=[]
                infer_image = batch["image"].to(device)

                for i in range(5):

                    batch["pred"] = sliding_window_inference(infer_image, (128,128,128), 1, models[i], mode='gaussian', overlap=0.5)

                    # inserito
                    val_outputs_list = decollate_batch(batch)
                    val_output_convert = [
                        post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
                    ]
                    val_output=val_output_convert[0]["pred_final"]

                    val_outputs_all.append(val_output)
                    out_to_save=nb.Nifti1Image(val_output.detach().cpu()[0, :, :, :].numpy(), img.affine, img.header)

                    save_output = join(work_dir, name_folder[i])
                    save_name_output= patient_id + name_file

                    if not os.path.exists(save_output):
                        os.makedirs(save_output)

                    nb.save(out_to_save, join(save_output,save_name_output))

                # ensemble
                summed_voxel=val_outputs_all[0]

                for i in range(1, 5):
                    summed_voxel=torch.add(summed_voxel, val_outputs_all[i])

                final_ensemble=torch.where(summed_voxel>2, 1, 0)
                ensemble_to_save=nb.Nifti1Image(final_ensemble.detach().cpu()[0, :, :, :].numpy(), img.affine, img.header)
                save_output_ensemble = join(work_dir, "ensemble_prediction")
                save_name_ensemble_output= patient_id + "ensemble_seg.nii.gz"
                
                if not os.path.exists(save_output_ensemble):
                    os.makedirs(save_output_ensemble)

                nb.save(ensemble_to_save, join(save_output_ensemble, save_name_ensemble_output))
        
        self.Prediction_path=save_output_ensemble

        return self.Prediction_path


class MyAlgoEnsembleBestN(AlgoEnsemble):
    """
    Ensemble method that select N model out of all using the models' best_metric scores

    Args:
        n_best: number of models to pick for ensemble (N).
    """

    def __init__(self, n_best: int = 5):

        super().__init__()
        self.n_best = n_best

    def sort_score(self):
        """
        Sort the best_metrics
        """
        scores = concat_val_to_np(self.algos, [AlgoEnsembleKeys.SCORE])
        return np.argsort(scores).tolist()

    def collect_algos(self, n_best: int = -1):
        """
        Rank the algos by finding the top N (n_best) validation scores.
        """

        if n_best <= 0:
            n_best = self.n_best

        ranks = self.sort_score()
        if len(ranks) < n_best:
            raise ValueError("Number of available algos is less than user-defined N")

        # get the indices that the rank is lower than N-n_best
        indices = [r for (i, r) in enumerate(ranks) if i < (len(ranks) - n_best)]

        # remove the found indices
        indices = sorted(indices, reverse=True)

        self.algo_ensemble = deepcopy(self.algos)

        for idx in indices:
            if idx < len(self.algo_ensemble):
                self.algo_ensemble.pop(idx)



class Predict_Finetuning:
    # Code to predict the Choroid Plexus segmentations taking finetuned models

     # initialization
    def __init__(self, work_dir: str=".", dataroot: str = ".", json_file: str = ".", output_dir=None, finetuning_dir=None):

        self.Workdir=work_dir
        self.Dataroot=dataroot
        self.JSON_file=json_file
        if finetuning_dir is None:
            self.Output_models_dir=join(self.Workdir, 'working_directory_finetuning')
        elif  isinstance(finetuning_dir, str):
            self.Output_models_dir=finetuning_dir
        if output_dir is None:
            self.Output_dir=join(self.Workdir, 'working_directory_prediction_finetuning')
        elif  isinstance(output_dir, str):
            self.Output_dir=output_dir
        self.Prediction_path=[]
       
    def predict(self):

        dataroot = self.Dataroot
        work_dir = self.Output_models_dir

        # create working directory
        if not os.path.isdir(work_dir):
            os.makedirs(work_dir)

        # write to a json file
        datalist = self.JSON_file

        # 1. Analyze Dataset

        data_src_cfg = join(work_dir, "data_src_cfg.yaml")

        data_src = {
        "modality": "MRI",
        "datalist": datalist,
        "dataroot": dataroot,
        }

        ConfigParser.export_config_file(data_src, data_src_cfg)

        # 2. Ensemble and save predictions

        pred_params = {
            'mode': "vote",   # use major voting 
            'sigmoid': False,
        }

        output_dir = join(self.Output_dir, "ensemble_prediction")
        self.Prediction_path=output_dir
        
        # create output directory
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        save_image = set_image_save_transform(output_dir, "ensemble_seg") # "ensemble" for major voting
        history=import_bundle_algo_history(work_dir, only_trained=False)
        builder = AlgoEnsembleBuilder(history, data_src_cfg)
        builder.set_ensemble_method(MyAlgoEnsembleBestN(n_best=5))
        ensembler = builder.get_ensemble()
        preds = ensembler(pred_param=pred_params)  
        print("Ensembling picked the following networks to ensemble:")
        for algo in ensembler.get_algo_ensemble():
            print(algo[AlgoEnsembleKeys.ID])

        for pred in preds:
            save_image(pred)
        logger.info(f"Ensemble prediction outputs are saved in {output_dir}.")

        return self.Prediction_path

