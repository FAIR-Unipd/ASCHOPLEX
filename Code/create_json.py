import os
import json
from collections import OrderedDict

join = os.path.join

class WriteJSON:
    # Writing .json files to run finetuning and/or the preidction of Choroid Plexus segmentations
    # Based on the parser flag you set, it will create different combinations of 

    # initialization
    def __init__(self, dataroot: str=".", description=None, work_dir: str=".", finetune: str=".", prediction: str="."):
   
        self.Dataroot=dataroot
        if description is None:
            self.Description='Dataset for Choroid Plexus segmentation'
        elif isinstance(description, str):
            self.Description=description
        self.JSON_dir=work_dir
        self.Finetune=finetune
        self.Prediction=prediction
        self.File=[]
       
    def write_json_file(self):
        
        # set data path
        output_folder = join(self.JSON_dir, 'JSON_file')
        
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        if self.Finetune=='yes'and self.Prediction=='yes':
            train_id=True
            test_id=True
            name_json="dataset_finetuning_prediction.json"

        elif (self.Finetune=='no' and self.Prediction=='yes') or (self.Finetune=='no' and self.Prediction=='ft'):
            train_id=False
            test_id=True
            name_json="dataset_prediction.json"
            
        # elif self.Finetune=='no' and self.Prediction=='ft':
        #     train_id=False
        #     test_id=True
        #     name_json="dataset_prediction_post_finetuning.json"

        else: 
            # self.Finetune=='yes' & self.Prediction=='no'
            train_id=True
            test_id=False
            name_json="dataset_finetuning.json"


        if train_id:
            train_dir = join(self.Dataroot, 'image_Tr')
            label_dir = join(self.Dataroot, 'label_Tr')
            train_ids=[]
            validation_ids=[]
            label_train_ids = []
            label_valid_ids=[]

            filenames_image = os.listdir(train_dir)
            filenames_image.sort()
            filenames_label = os.listdir(label_dir)
            filenames_label.sort()           

            # training

            for name in filenames_image[0:5]:
                if not(name.endswith('.nii') | name.endswith('.nii.gz')):
                    raise ValueError("Data are not in the correct format. Please, provide images in .nii or .nii.gz Nifti format")
                image=join(train_dir, name)
                train_ids.append(image)
            
            for name in filenames_label[0:5]:
                if not(name.endswith('.nii') | name.endswith('.nii.gz')):
                    raise ValueError("Data are not in the correct format. Please, provide images in .nii or .nii.gz Nifti format")
                label=join(label_dir, name)
                label_train_ids.append(label)
            
            # validation

            for name in filenames_image[5:10]:
                if not(name.endswith('.nii') | name.endswith('.nii.gz')):
                    raise ValueError("Data are not in the correct format. Please, provide images in .nii or .nii.gz Nifti format")
                image=join(train_dir, name)
                validation_ids.append(image)
            
            for name in filenames_label[5:10]:
                if not(name.endswith('.nii') | name.endswith('.nii.gz')):
                    raise ValueError("Data are not in the correct format. Please, provide images in .nii or .nii.gz Nifti format")
                label=join(label_dir, name)
                label_valid_ids.append(label)

        if test_id:
            #  testing
            
            test_dir=join(self.Dataroot, 'image_Ts')
            test_ids=[]
            testnames = os.listdir(test_dir)
            testnames.sort()

            for test_name in testnames:
                if not(test_name.endswith('.nii') | test_name.endswith('.nii.gz')):
                    raise ValueError("Data are not in the correct format. Please, provide images in .nii or .nii.gz Nifti format")
                image=join(test_dir, test_name)
                test_ids.append(image)


        # manually set
        json_dict = OrderedDict()
        json_dict['name'] = "MRI Dataset - Choroid Plexus Segmentation" 
        json_dict['description'] = self.Description
        json_dict['tensorImageSize'] = "3D"
        json_dict['modality'] = {
            "0": "MR"
        }
        # manually set
        json_dict['labels'] = {
            "0": "background",
            "1": "Choroid Plexus"
        }

        if train_id and test_id:

            json_dict['numTraining'] = len(train_ids)
            json_dict['numValidation'] = len(validation_ids)
            json_dict['numTest'] = len(test_ids)
            json_dict['training'] = [{"fold": 0, "image": '%s' %i , "label": '%s' %j} for j, i in zip(label_train_ids, train_ids)]
            json_dict['validation'] = [{"image": '%s' %i, "label": '%s' %j} for j,i in zip(label_valid_ids, validation_ids)]
            json_dict['testing'] = [{"image": '%s' %i} for i in test_ids]

        elif not(train_id) and test_id: 

            json_dict['numTest'] = len(test_ids)
            json_dict['testing'] = [{"image": '%s' % i} for i in test_ids]

        else:

            # train_id and not(test_id)
            json_dict['numTraining'] = len(train_ids)
            json_dict['numValidation'] = len(validation_ids)
            json_dict['training'] = [{"fold": 0, "image": '%s' %i , "label": '%s' %j} for j, i in zip(label_train_ids, train_ids)]
            json_dict['validation'] = [{"image": '%s' %i, "label": '%s' %j} for j,i in zip(label_valid_ids, validation_ids)]

        with open(join(output_folder, name_json), 'w') as f:
            json.dump(json_dict, f, indent=4, sort_keys=True)

        self.File.append(join(output_folder, name_json))

        return self.File[0] 

