import csv

with open('../tuhoi_dataset.csv', errors='ignore') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        '''print(row[18] + ":\n")
        print(row[6] + "\n")

        print(row[21] + ":\n")
        print(row[8] + "\n")'''

        print('Reading line ' + str(line_count) + '...\n')
        for obj in row[6].split('\n'):
            print('Objeto: ' + row[18])
            print(obj + " in " + row[6] + '\n')
        
        line_count+=1
        if line_count >= 10764:
            break

    print(f'Processed {line_count} lines.')