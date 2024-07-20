import csv

def add_length_to_metadata():
    short = ['1', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101','102','103','104','105','106','107','108','109','110','111','112', '113', '114']
    medium_short = ['13', '14', '15', '30', '31', '32', '44', '45', '46', '47', '48', '49', '50' , '51', '52', '53', '54', '55', '56', '57', '58', '59', '60']
    medium = ['8', '10', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43']
    medium_long = ['9', '11', '12', '16']
    long = ['2', '3', '4', '5', '6', '7']
    
    meta_path = f'./train/metadata/metadata.csv'
    
    with open(f'./train/metadata/expanded_metadata.csv', 'w+') as newfile:
        newfile.write('folder,chapter num,chapter name,reciter id,length\n')
        with open(meta_path, 'r+') as meta:
            # skip first line
            next(meta)
            meta_reader = csv.reader(meta, delimiter=',')
            val = ''
            for line in meta_reader:
                if line[1] in short:
                    val = 'short'
                if line[1] in medium_short:
                    val = 'medium short'
                if line[1] in medium:
                    val = 'medium'
                if line[1] in medium_long:
                    val = 'medium long'
                if line[1] in long:
                    val = 'long'
                newfile.write(','.join(line)+f',{val}\n')
                
            
if __name__ == '__main__':
    add_length_to_metadata()
