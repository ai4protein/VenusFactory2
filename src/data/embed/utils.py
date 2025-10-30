import os


def read_multi_fasta(file_path):
    data = []
    # if file_path is a directory, read all fasta files in the directory
    if os.path.isdir(file_path):
        for file in os.listdir(file_path):
            if file.endswith('.fasta'):
                data.extend(read_multi_fasta(os.path.join(file_path, file)))
        return data
    
    # if file_path is a file, read the file
    else:
        current_sequence = ''
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('>'):
                    if current_sequence:
                        data.append({"header": header, "sequence": current_sequence})
                        current_sequence = ''
                    header = line[1:]
                else:
                    current_sequence += line
            if current_sequence:
                data.append({"header": header, "sequence": current_sequence})
        return data