import argparse
from data_and_models.models import train_model, predict, demo, device_options
from data_and_models.dataset import color

if __name__ == '__main__':
    # // commands //
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--predict', action='store_true', help='Predict mode.')
    parser.add_argument('--train', action='store_true', help='Train mode.')
    parser.add_argument('--demo', action='store_true', help='Test-Set Demo.')
    parser.add_argument('-t', '--target', type=str, choices=('reciter'), required=False, default='all', help='Prediction target: "reciter", "chapter", or "all".')
    parser.add_argument('-i', '--input', type=str, required=False, help='Audio file input for prediction / inference.')
    parser.add_argument('-dev', '--device', type=str, choices=device_options, default='cpu', required=False, help='Device selection for training and inference / predicting.')
    parser.add_argument('-m', '--model', type=str, choices=('cnn_reciter'), required=False, help='Model selection for training: "cnn_reciter" or "cnn_chapter".')
    args = parser.parse_args()
    
    # demo
    if args.demo == True and args.predict == False and args.train == False:
        demo(target_type='reciter')
    
    # predict
    if args.predict == True and args.train == False and args.demo == False:
        if args.target == 'reciter':
            predict(target_type='reciter', device=args.device, input_file=args.input)
        # elif args.category == 'chapter':
        #     predict(target='chapter', device=args.device, input_file=args.input)
        # elif args.target == 'all':
        #     predict(target_type='reciter', device=args.device, input_file=args.input)
        #     predict(target_type='chapter', device=args.device, input_file=args.input)
        else:
            print(f'\n{color.RED}Invalid or missing target (-t)!{color.END}')
            
    # train      
    elif args.train == True and args.predict == False and args.demo == False and args.model != None:
        if args.model == 'cnn_reciter':
            train_model(target_type='reciter', device=args.device)
        # elif args.model == 'cnn_chapter':
        #     train_model(target_type='chapter', device=args.device)
    elif args.train == False and args.predict == False and args.demo == False:
        print(f'\n{color.RED}Specify mode --predict or --train!{color.END}')
    elif args.model == None and args.train == True:    
        print(f'\n{color.RED}Please choose a model to train via command -m (--model)!{color.END}')
    elif args.model != None and args.train == False and args.predict == True:
        print(f'\n{color.RED}Command -m (--model) is only for --train mode!{color.END}')
