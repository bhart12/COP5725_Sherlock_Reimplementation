import os

from google_drive_downloader import GoogleDriveDownloader as gd

#Download raw and preprocessed data if they do no already exist
data_dir = './data.zip'
print(f"Downloading the raw and preprocessed data into {data_dir}.")

if not os.path.exists(data_dir):
    print('Downloading data directory.')
    dir_name = data_dir
    gd.download_file_from_google_drive(
        file_id='1-g0zbKFAXz7zKZc0Dnh74uDBpZCv4YqU',
        dest_path=dir_name,
        unzip=True,
        showsize=True
    )

print('Data was downloaded.')






#Download data files for feature extraction information if they do no already exist
word_embedding_file = '../sherlock/features/glove.6B.50d.txt'
paragraph_vector_file = '../sherlock/features/par_vec_trained_400.pkl.docvecs.vectors_docs.npy'

print(
    f"""Preparing feature extraction by downloading 2 files:
    \n {word_embedding_file} and \n {paragraph_vector_file}.
    """
)

if not os.path.exists(word_embedding_file):
    print('Downloading GloVe word embedding vectors.')
    file_name = word_embedding_file
    gd.download_file_from_google_drive(
        file_id='1kayd5oNRQm8-NCvA8pIrtezbQ-B1_Vmk',
        dest_path=file_name,
        unzip=False,
        showsize=True
    )

    print("GloVe word embedding vectors were downloaded.")

if not os.path.exists(paragraph_vector_file):
    print("Downloading pretrained paragraph vectors.")
    file_name = paragraph_vector_file
    gd.download_file_from_google_drive(
        file_id='1vdyGJ4aB71FCaNqJKYX387eVufcH4SAu',
        dest_path=file_name,
        unzip=False,
        showsize=True
    )

    print("Trained paragraph vector model was downloaded.")

print("All files for extracting word and paragraph embeddings are present.")
