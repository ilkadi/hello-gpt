import os
import argparse
import tiktoken
import numpy as np

class TxtToTokenConverter:
    def __init__(self):
        self.encoding = tiktoken.get_encoding("gpt2")
    
    def convert(self, input_path, output_dir_path):      
        if os.path.isdir(input_path):
            print("Converting directory..")
            data = ""
            for filename in os.listdir(input_path):
                if filename.endswith('.txt'):
                    print(f"\tReading file {filename}..")
                    with open(os.path.join(input_path, filename), 'r', encoding='utf-8') as f:
                        data += f.read()
        else:  
            print("Converting file..")
            with open(input_path, 'r', encoding='utf-8') as f:
                data = f.read()
                
        n = len(data)
        train_data = data[:int(n*0.9)]
        val_data = data[int(n*0.9):]

        train_ids = self.encoding.encode_ordinary(train_data)
        val_ids = self.encoding.encode_ordinary(val_data)
        print(f"Train set: {len(train_ids):,} tokens")
        print(f"Validation set: {len(val_ids):,} tokens")

        train_ids = np.array(train_ids, dtype=np.uint16)
        val_ids = np.array(val_ids, dtype=np.uint16)
        train_ids.tofile(os.path.join(output_dir_path, 'train.bin'))
        val_ids.tofile(os.path.join(output_dir_path, 'validate.bin'))
        
def main(input_file_path, directory_path):
    TxtToTokenConverter().convert(input_file_path, directory_path)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start the convertsion from txt to tokens.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to txt file or directory with txt files.')
    parser.add_argument('--output_dir_path', type=str, required=True, help='Path to directory.')
    args = parser.parse_args()

    main(args.input_path, args.output_dir_path)