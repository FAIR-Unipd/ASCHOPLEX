import os
import argparse
import sys

from Code.create_json import WriteJSON
from Code.prediction import Predict_Directly, Predict_Finetuning
from Code.finetuning import Finetuning

join = os.path.join

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

# Main
if __name__ == '__main__':

    # Initialize the parser
    parser = argparse.ArgumentParser(
        description="Pipeline for finetuning with N subjects with selected pretrained models"
    )

    # Add the parameters positional/optional
    parser.add_argument('--dataroot', required=True, help="Data directory")
    parser.add_argument('--description', required=False, help="Data description")
    parser.add_argument('--work_dir', required=True, help="working directory")
    parser.add_argument('--finetune', required=True, choices=['yes', 'no'], help="finetuning flag")
    parser.add_argument('--prediction', required=True, choices=['yes', 'no', 'ft'], help="prediction flag")
    parser.add_argument('--output_pred_dir', required=False, help="Working directory where to save predictions. If not specified, default folder name and locations will be used")
    parser.add_argument('--finetune_dir', required=False, help="Working directory where to save finetuned models. If not specified, default folder name and locations will be used")

    # Parse the arguments
    args = parser.parse_args()
    print(args)

    if (args.finetune == 'yes' and args.prediction == 'ft'):
        raise ValueError("You have already run the finetuning step. You can use it to obtain Choroid Plexus segmentations. Please, if you wanna set --prediction ft, you must set --finetune no")
    if (args.finetune == 'no' and args.prediction == 'no'):
        raise ValueError("You are doing nothing with this parser combination. Please, decide if you wanna run the finetuning, only the prediction, or both")
    
    print('---------------ASCHOPLEX PIPELINE IS STARTING----------------')
    
    print('Writing JSON file.....')
    # json_file=WriteJSON(args.dataroot, args.description, args.work_dir, args.finetune, args.prediction).write_json_file()
    json_file=WriteJSON(args.dataroot, args.description, args.work_dir, args.finetune, args.prediction).write_json_file()

    print('JSON file: DONE')
    print('Your JSON file has been created here: ', json_file)

    if args.finetune == 'yes' and args.prediction == 'yes':

        print('You choose to run the finetuning ASCHOPLEX pipeline and to obtain the predictions')
        print('Finetuning starting.....')

        finetuning_directory=Finetuning(args.work_dir, args.dataroot, json_file, args.finetune_dir).finetuning_run()
        
        print('Your finetuned models are saved here: ', finetuning_directory)

        print('Finetuning: DONE')
        print('Prediction starting.....')

        output_ensemble_directory=Predict_Finetuning(args.work_dir, args.dataroot, json_file, args.output_pred_dir, args.finetune_dir).predict()

        print('Prediction: DONE')
        print('Your Ensemble Predictions are saved here: ', output_ensemble_directory)

    elif args.finetune == 'no' and args.prediction == 'yes':

        print('You choose to have directly the prediction. ATTENTION! This is not the suggested choice: your dataset features would not match with that learnt by ASCHOPLEX')
        
        print('Ensemble Prediction starting.....')

        output_ensemble_directory=Predict_Directly(args.work_dir, json_file, args.output_pred_dir).predict()

        print('Ensemble Prediction: DONE')
        print('Your Ensemble Predictions are saved here: ', output_ensemble_directory)
    
    elif args.finetune == 'yes' and args.prediction == 'no':

        print('You choose to run the finetuning ASCHOPLEX pipeline but not to obtain the predictions at the same time')
        print('Finetuning starting.....')

        finetuning_directory=Finetuning(args.work_dir, args.dataroot, json_file, args.finetune_dir).finetuning_run()
        
        print('Your finetuned models are saved here: ', finetuning_directory)
        print('Finetuning: DONE')
    
    else:
        # args.finetune == 'no' and args.prediction == 'ft'
        print('You choose to obtain the predictions with the finetuned ASCHOPLEX models run in a previous time')
        
        print('Prediction starting.....')

        output_ensemble_directory=Predict_Finetuning(args.work_dir, args.dataroot, json_file, args.output_pred_dir, args.finetune_dir).predict()

        print('Prediction: DONE')
        print('Your Ensemble Predictions are saved here: ', output_ensemble_directory)


    print('---------------ASCHOPLEX PIPELINE IS FINISHED----------------')








