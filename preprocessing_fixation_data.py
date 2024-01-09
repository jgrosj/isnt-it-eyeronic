import csv

# set cognitive data file path
cognitive_data_source_path = "define path""¨

# set up dictionaries
word_id_and_word_dict = {}
word_id_and_fixation_duration_dict = {}
text_id_and_word_id_and_fixation_duration_dict = {}
output_sequence_add_ons = {}

# open csv file and read lines
with open(cognitive_data_source_path, 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    
    # skip header
    next(csv_reader)

    # iterate over lines
    for line in csv_reader:

        # skip aspect and empty values
        if "Aspect" in line[3]:
            continue
        
        if line[3] == "0":
            continue

        if line[3] == "#N/A":
            continue

        # get text id
        text_id = line[1]

        # get word ids and words
        word_id = text_id + line[2]
        word = line[3]

        # remove "ﾒ" from words
        if "ﾒ" in word:
            word = word.replace("ﾒ", "")

        # add word id and word to dictionary
        if word_id not in word_id_and_word_dict:
            word_id_and_word_dict[word_id] = word

        # get fixation durations
        fixation_duration = int(line[4])

        # add word id and fixation duration to dictionary
        if word_id not in word_id_and_fixation_duration_dict:
            word_id_and_fixation_duration_dict[word_id] = fixation_duration
        else:
            word_id_and_fixation_duration_dict[word_id] += fixation_duration

        # add text id and word id and fixation duration to dictionary
        if text_id not in text_id_and_word_id_and_fixation_duration_dict:
            text_id_and_word_id_and_fixation_duration_dict[text_id] = {}
        text_id_and_word_id_and_fixation_duration_dict[text_id][word_id] = word_id_and_fixation_duration_dict[word_id]

# iterate over text ids
for text_id, word_id_and_fixation_duration_dict in text_id_and_word_id_and_fixation_duration_dict.items():

    # order word ids by fixation duration from highest to lowest
    word_ids_ordered_by_fixation_duration = sorted(word_id_and_fixation_duration_dict, key=lambda x: word_id_and_fixation_duration_dict[x], reverse=True)

    # iterate over word ids
    output_sequence_add_ons[text_id] = [word_id_and_word_dict[word_id] for word_id in word_ids_ordered_by_fixation_duration]

# set sequence data path
sequence_data_path = "define path"

# output path for the first new CSV file with the sequence and add-ons
output_csv_path1 = "define path"

# output path for the second new CSV file with the sequences twice (baseline)
output_csv_path2 = "define path"

# create dictionaries with id and sequence
sequence_dict = {}
label_dict = {}

# open sequence data file and read lines
with open(sequence_data_path, 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)

    # skip header
    next(csv_reader)
    
    # iterate over lines
    for line in csv_reader:
        # get id and sequence
        id = line[0]
        sequence = line[1]
        # get label
        label = line[5]
        # add id and sequence to dictionary
        sequence_dict[id] = sequence
        # add id and label to dictionary
        label_dict[id] = label

# write enhanced input data file
with open(output_csv_path1, 'w', encoding='utf-8', newline='') as output_file:
    csv_writer = csv.writer(output_file)

    # write header
    csv_writer.writerow(["Text ID", "Sequence", "Label"])

    # write data rows
    for text_id in sequence_dict:
        csv_writer.writerow([text_id, sequence_dict[text_id] + " [SEP] " + " ".join(output_sequence_add_ons[text_id]), label_dict[text_id]])

# write baseline data file
with open(output_csv_path2, 'w', encoding='utf-8', newline='') as output_file:
    csv_writer = csv.writer(output_file)

    # write header
    csv_writer.writerow(["Text ID", "Sequence", "Label"])

    # write data rows
    for text_id in sequence_dict:
        csv_writer.writerow([text_id, sequence_dict[text_id] + " [SEP] " + sequence_dict[text_id], label_dict[text_id]])
