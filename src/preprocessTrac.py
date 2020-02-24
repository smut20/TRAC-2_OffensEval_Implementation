import re, csv, random, html
from utils import save_obj
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 50


def create_data_task(train_file, vocab_path, word_index_pkl_path, embedding_matrix_path, emb_path, data_path,
                      labels_path, word_index_path, nlp):
    # WORD_INDEX
    word_index = get_word_index_CSV(vocab_path, nlp)

    out = open(word_index_path, "w", encoding='utf-8')
    out.write("word" + "\t" + "index" + "\n")
    for k in sorted(word_index):
        out.write(k + "\t" + str(word_index[k]) + "\n")
    out.close()
    save_obj(word_index, word_index_pkl_path)

    # embedding_matrix
    embedding_matrix = get_pretrained_embedding(emb_path, word_index)
    save_obj(embedding_matrix, embedding_matrix_path)

    label_dict = {"OAG": 0,
                  "CAG": 1,
                  "NAG": 2}
    data, labels = get_preprocessed_tweets(train_file, nlp, word_index, "subtask_a", label_dict)  # ?
    save_obj(data, data_path)
    save_obj(labels, labels_path)


def prepocess_organizers_dataset(add_path, add_proc_path):
    count = 0
    total = 0

    out = open(add_proc_path, "w", encoding='utf-8')
    out.write("id\ttext\tsubtask_a\tsubtask_b\n")

    with open(add_path, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            textColumn = row['Text'].replace("\n", " ")

            total += 1
            if len(textColumn.split(" ")) > 55:
                continue
            count += 1

            textColumn = re.sub(r"http\S+", "URL", textColumn)
            textColumn = re.sub('@[^\s]+', '@USER', textColumn)

            outputText = row['ID'] + "\t" + textColumn + "\t" + row['Sub-task A'] + "\t" + row['Sub-task B'] + "\n"
            out.write(outputText)

    print("Preprocessing finished")
    out.close()


def prepocess_additional_dataset(add_path, add_proc_path):
    count = 0
    total = 0

    out = open(add_proc_path, "w", encoding='utf-8')
    with open(add_path, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            columnText = row['text'].replace("\n", " ")
            total += 1
            if columnText.startswith("RT"):
                continue
            if " RT " in columnText:
                continue
            if not len(columnText):
                continue
            count += 1

            columnText = re.sub(r"http\S+", "URL", columnText)
            columnText = re.sub('@[^\s]+', '@USER', columnText)

            outputText = row['id'] + "\t" + columnText + "\t" + row['class'] + "\n"
            out.write(outputText)

    print(total)
    print(count)
    out.close()


def preprocess_task3(train_file, new_path):
    out = open(new_path, "w", encoding="utf-8")
    original_train = []
    duplicated_train = []
    fd = open(train_file, "r", encoding="utf-8")
    read = csv.DictReader(fd, dialect="excel-tab")
    for row in read:
        nl = row["id"] + "\t" + row["text"] + "\t" + row["subtask_a"] + "\t" + row["subtask_b"] + "\t"
        original_train.append(nl)

    random.shuffle(original_train)

    test_count = int(len(original_train) * 0.15)
    print("test_count: " + str(test_count))
    print("duplicated_train: " + str(len(duplicated_train)))

    test_data = original_train[:test_count]
    original_train = original_train[test_count:]

    perfect_duplicated_data = []

    print("test_data: " + str(len(test_data)))
    print("original_train: " + str(len(original_train)))

    for l in duplicated_train:
        if not l in test_data:
            perfect_duplicated_data.append(l)

    print("perfect_duplicated_data: " + str(len(perfect_duplicated_data)))

    original_train = original_train + perfect_duplicated_data + perfect_duplicated_data + perfect_duplicated_data

    random.shuffle(original_train)
    random.shuffle(original_train)

    original_train = test_data + original_train

    out.write(
        "id" + "\t" + "text" + "\t" + "subtask_a" + "\t" + "subtask_b" + "\n")  # here I need to delete the last subtask

    for nl in original_train:
        out.write(nl + "\n")

    out.close()
    fd.close()


def get_pretrained_embedding(emb_path, word_index):
    embeddings_index = {}
    f = open(emb_path, encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.random.uniform(-0.8, 0.8, (len(word_index) + 1, EMBEDDING_DIM))

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def clean_tweet(tweet_text):
    tweet_text = tweet_text.replace('—', " ")  # .replace("'", "’")
    tweet_text = ' '.join(tweet_text.split())
    return tweet_text.strip()


def text_processing(nlp, text):
    tweet_text = clean_tweet(text)

    tweet_text = tweet_text.replace("'", "&&&").replace("’", "&&&").replace("-", "&#&")

    doc = nlp(tweet_text)
    print(text)

    for token in doc:
        print(str(token).replace("&&&", "'").replace("&#&", "-"))


def replace_tweet(tweet_text):
    tweet_text = html.unescape(tweet_text)

    return tweet_text.replace("'", "QUOTE_SYMBOL").replace("‘", "QUOTE_SYMBOL").replace("’", "QUOTE_SYMBOL").replace(
        "-", "HYPH_SYMBOL").replace(";", " ").replace("#", "HASHTAG_SYMBOL")


def unreplace_tweet(tweet_text):
    return tweet_text.replace("QUOTE_SYMBOL", "'").replace("HYPH_SYMBOL", "-").replace("HASHTAG_SYMBOL", "#").replace(
        "EMOJI_SYMBOL", "#&").lower()


def get_preprocessed_labels(path, label_type="subtask_a", label_dict={"OFF": 1}):  # Here I changed "OFF" to
    fd = open(path, "r", encoding="utf-8")
    read = csv.DictReader(fd, dialect="excel-tab")
    labels = []

    for row in read:
        label = label_dict.get(row[label_type], 0)
        labels.append(label)

    return labels


def get_word_index_CSV(path, nlp):
    fd = open(path, "r", encoding="utf-8")

    word_set = set()
    i = 0

    for row in csv.reader(fd, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
        if not i:
            i += 1
            continue
        i += 1
        tweet_text = row[1].replace("\n", " ").replace("\r", " ")
        tweet_text = re.sub(' +', ' ', tweet_text)
        tweet_text = clean_tweet(tweet_text)
        tweet_text = replace_tweet(tweet_text)
        doc = nlp(tweet_text)
        for token in doc:
            word = unreplace_tweet(str(token))
            word_set.add(word)

    word_index = {}
    i = 0
    for tok in word_set:
        i += 1
        word_index[tok] = i

    fd.close()
    return word_index


def get_word_index_from_file(path):
    fd = open(path, "r")

    read = csv.DictReader(fd, dialect="excel-tab")
    word_index = {}
    for row in read:
        word_index[row["word"]] = int(row["index"])
    fd.close()
    return word_index


def get_preprocessed_tweets(path, nlp, word_index, label_type="subtask_a", label_dict={"OFF": 1}, max_instances=0):
    print("Processing file " + path)

    fd = open(path, "r", encoding="utf-8")
    read = csv.DictReader(fd, dialect="excel-tab")
    texts = []
    labels = []
    i = 0
    for row in read:
        sent = []
        i += 1
        tweet_text = clean_tweet(row["text"])
        if "test" in label_type:
            label = row["id"]
        else:
            label = label_dict.get(row[label_type], 0)
        labels.append(label)
        tweet_text = replace_tweet(tweet_text)

        doc = nlp(tweet_text)

        for token in doc:
            word = unreplace_tweet(str(token))
            sent.append(word)

        texts.append(sent)
        if i == max_instances:
            break

    sequences = []
    for sent in texts:
        seq = []
        for word in sent:
            i = word_index.get(word, 0)
            if i:
                seq.append(i)
        sequences.append(seq)

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    if not "test" in label_type:
        labels = to_categorical(np.asarray(labels))
    fd.close()
    return data, labels


def create_test_data(data_path, ids_test_path, testing_path, word_index, nlp):
    data_test, ids_test = get_preprocessed_tweets(testing_path, nlp, word_index, "test")
    save_obj(data_test, data_path)
    save_obj(ids_test, ids_test_path)
