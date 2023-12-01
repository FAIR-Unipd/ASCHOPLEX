import os
import json
import math
from collections import OrderedDict

join = os.path.join

class WriteJSON:
    # Writing .json files to run finetuning and/or the prediction of Choroid Plexus segmentations

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

        if self.Finetune=='yes':
        
            train_id=True

            if self.Prediction=='yes':

                test_id=True
                test_ft=False

            else: 
                # self.Prediction=='no'
                test_id=False
                test_ft=False
                
            name_json="dataset_finetuning.json"

        else:

            train_id=False

            if self.Prediction=='yes':
                
                test_id=True
                test_ft=False
                name_json="dataset_prediction.json"

            else:
                # self.Prediction=='ft'
                test_id=False
                test_ft=True
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

            if len(filenames_image)!=len(filenames_label):
                raise ValueError("The number of images and the number of labels is different. Please, check image_Tr and label_Tr folders.")

            # training
            jj=math.ceil(len(filenames_image)/2)

            for name in filenames_image[0:jj]:
                if not(name.endswith('.nii') | name.endswith('.nii.gz')):
                    raise ValueError("Data are not in the correct format. Please, provide images in .nii or .nii.gz Nifti format")
                image=join(train_dir, name)
                train_ids.append(image)
            
            count=0
            for name in filenames_label[0:jj]:
                if not(name.endswith('.nii') | name.endswith('.nii.gz')):
                    raise ValueError("Data are not in the correct format. Please, provide images in .nii or .nii.gz Nifti format")
                img_=os.path.basename(filenames_image[count]).replace('_image', '')
                lab_=os.path.basename(name).replace('_seg', '')
                if img_==lab_:
                    label=join(label_dir, name)
                    label_train_ids.append(label)
                    count+=1
                else:
                    raise ValueError("Subject identifier is not univoque. Please, pass correct data")
            
            # validation

            for name in filenames_image[jj:len(filenames_image)]:
                if not(name.endswith('.nii') | name.endswith('.nii.gz')):
                    raise ValueError("Data are not in the correct format. Please, provide images in .nii or .nii.gz Nifti format")
                image=join(train_dir, name)
                validation_ids.append(image)
            count=jj
            for name in filenames_label[jj:len(filenames_image)]:
                if not(name.endswith('.nii') | name.endswith('.nii.gz')):
                    raise ValueError("Data are not in the correct format. Please, provide images in .nii or .nii.gz Nifti format")
                img_=os.path.basename(filenames_image[count]).replace('_image', '')
                lab_=os.path.basename(name).replace('_seg', '')
                if img_==lab_:
                    label=join(label_dir, name)
                    label_valid_ids.append(label)
                    count+=1
                else:
                    raise ValueError("Subject identifier is not univoque. Please, pass correct data")


        if test_id or test_ft:
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


        if (train_id and test_id) or (not(train_id) and test_id) or (train_id and not(test_id)):

            # create json file - manually set

            json_dict = OrderedDict()
            json_dict['name'] = "MRI Dataset - Choroid Plexus Segmentation" 
            json_dict['description'] = self.Description
            json_dict['tensorImageSize'] = "3D"
            json_dict['modality'] = {
                "0": "MR"
            }
            
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

            elif train_id and not(test_id):

                json_dict['numTraining'] = len(train_ids)
                json_dict['numValidation'] = len(validation_ids)
                json_dict['training'] = [{"fold": 0, "image": '%s' %i , "label": '%s' %j} for j, i in zip(label_train_ids, train_ids)]
                json_dict['validation'] = [{"image": '%s' %i, "label": '%s' %j} for j,i in zip(label_valid_ids, validation_ids)]

            with open(join(output_folder, name_json), 'w') as f:
                json.dump(json_dict, f, indent=4, sort_keys=True)
        
        elif test_ft:

            # append lines to the .json files
            with open(join(output_folder, name_json)) as f:
                json_append=json.load(f)

            json_append['numTest']=len(test_ids)

            json_append['testing'] = [{"image": '%s' % i} for i in test_ids]

            with open(join(output_folder, name_json), 'w') as f:
                json.dump(json_append, f, indent=4, sort_keys=True)

        self.File.append(join(output_folder, name_json))

        return self.File[0] 

