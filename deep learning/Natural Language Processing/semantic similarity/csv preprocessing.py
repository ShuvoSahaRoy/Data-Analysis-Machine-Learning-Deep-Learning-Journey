import csv,re

# with open('stsbenchmark/sts-train.csv', 'r', encoding = 'utf-8') as infile, open('repaired.csv','w', encoding='utf-8') as outfile:
#     for line in infile.readlines():
#         error = 0
#         try:
#             line = line.replace('","', '')
#             # line = line.replace('",', '')
#             outfile.write(line)
#         except:
#             error += 1
        
#     print(error)
    
with open('repaired.csv', 'r', encoding = 'utf-8-sig') as file:
    text1 = []
    text2 = []
    label = []
    reader = csv.reader(file)
    count=1
    error = 0
    for row in reader:
        first_element = row[0]
        # check.append(first_element)
        try:
            sent2 = first_element.split('\t')[6]
            sent1 = first_element.split('\t')[5]
            target = first_element.split('\t')[0]
            
            text2.append(sent2)
            text1.append(sent1)
            label.append(target)
        except:
            error+=1 
        
    print(error)
