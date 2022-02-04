
import os
import urllib.request
import zipfile


DATA_PATH = "../data/msmd.zip"
DATA_URL = "https://zenodo.org/record/4745838/files/msmd.zip?download=1"
if __name__ == '__main__':

    if not os.path.exists(DATA_PATH):

        if not os.path.exists(os.path.dirname(DATA_PATH)):
            print('Creating data folder ...')
            os.mkdir(os.path.dirname(DATA_PATH))

        print(f"Downloading data to {DATA_PATH} ...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)

        print(f"Extracting data {DATA_PATH} ...")
        zip_ref = zipfile.ZipFile(DATA_PATH, 'r',  zipfile.ZIP_DEFLATED)
        zip_ref.extractall(os.path.dirname(DATA_PATH))
        zip_ref.close()

        # delete zip file
        os.unlink(DATA_PATH)
