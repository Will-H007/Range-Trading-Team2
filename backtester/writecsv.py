import csv


class fetch_data:

    def __init__(self, input_data, output_csv_file):
        self.input_data = input_data
        self.output_csv_file = output_csv_file

    #Assume data is splited by comma
    #Store data into lists of list
    def read_data(self):
        with open(self.input_data, 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                lines[i] = lines[i].strip("\n").split(',')
        print("The file content are: ",lines)
        return lines

    def write_to_csv(self):
        # open the file in the write mode (newline = '' in order to)
        with open(f"{self.output_csv_file}.csv", 'w', newline='') as f:

            # create the csv writer
            writer = csv.writer(f)

            data_lst = self.read_data()

            for row in data_lst:
                # write each row to the csv file
                writer.writerow(row)

testing = fetch_data("testing.txt", "testing")
testing.read_data()
testing.write_to_csv()