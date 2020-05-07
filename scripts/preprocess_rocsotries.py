import csv


title = []
content = []
rocstories_location = '../data/ROCStories_winter2017 - ROCStories_winter2017.csv'
output_location = '../data/cleaned_rocstories.csv'

with open(rocstories_location, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        title.append(row['storytitle'])
        content.append(' '.join([row['sentence1'], row['sentence2'],
                              row['sentence3'], row['sentence4'],
                              row['sentence5']]))


with open(output_location, 'w', newline='') as csvfile:
    fieldnames = ['storytitle', 'content']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for t, c in zip(title, content):
        writer.writerow({'storytitle': t, 'content': c.strip('"')})


