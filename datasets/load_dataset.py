#   =====================================================================
#   Copyright (C) 2023  Stefan Schubert, stefan.schubert@etit.tu-chemnitz.de
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.
#   =====================================================================
#
import os
import urllib.request
import zipfile
from glob import glob
from PIL import Image
import numpy as np
from scipy.signal import convolve2d
from typing import List, Tuple
from abc import ABC, abstractmethod


class Dataset(ABC):
    @abstractmethod
    def load(self) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def download(self, destination: str):
        pass


class GardensPointDataset(Dataset):
    def __init__(self, destination: str = 'images/GardensPoint/'):
        self.destination = destination

    def load(self) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        print('===== Load dataset GardensPoint day_right--night_right')

        # download images if necessary
        if not os.path.exists(self.destination):
            self.download(self.destination)

        # load images
        fns_db = sorted(glob(self.destination + 'day_right/*.jpg'))
        fns_q = sorted(glob(self.destination + 'night_right/*.jpg'))

        imgs_db = [np.array(Image.open(fn)) for fn in fns_db]
        imgs_q = [np.array(Image.open(fn)) for fn in fns_q]

        # create ground truth
        GThard = np.eye(len(imgs_db)).astype('bool')
        GTsoft = convolve2d(GThard.astype('int'),
                            np.ones((17, 1), 'int'), mode='same').astype('bool')

        return imgs_db, imgs_q, GThard, GTsoft

    def download(self, destination: str):
        print('===== GardensPoint dataset does not exist. Download to ' + destination + '...')

        fn = 'GardensPoint_Walking.zip'
        url = 'https://www.tu-chemnitz.de/etit/proaut/datasets/' + fn

        # create folders
        path = os.path.expanduser(destination)
        os.makedirs(path, exist_ok=True)

        # download
        urllib.request.urlretrieve(url, path + fn)

        # unzip
        with zipfile.ZipFile(path + fn, 'r') as zip_ref:
            zip_ref.extractall(destination)

        # remove zipfile
        os.remove(destination + fn)


class StLuciaDataset(Dataset):
    def __init__(self, destination: str = 'images/StLucia_small/'):
        self.destination = destination

    def load(self) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        print('===== Load dataset StLucia 100909_0845--180809_1545 (small version)')

        # download images if necessary
        if not os.path.exists(self.destination):
            self.download(self.destination)

        # load images
        fns_db = sorted(glob(self.destination + '100909_0845/*.jpg'))
        fns_q = sorted(glob(self.destination + '180809_1545/*.jpg'))

        imgs_db = [np.array(Image.open(fn)) for fn in fns_db]
        imgs_q = [np.array(Image.open(fn)) for fn in fns_q]

        # create ground truth
        gt_data = np.load(self.destination + 'GT.npz')
        GThard = gt_data['GThard'].astype('bool')
        GTsoft = gt_data['GTsoft'].astype('bool')

        return imgs_db, imgs_q, GThard, GTsoft

    def download(self, destination: str):
        print('===== StLucia dataset does not exist. Download to ' + destination + '...')

        fn = 'StLucia_small.zip'
        url = 'https://www.tu-chemnitz.de/etit/proaut/datasets/' + fn

        # create folders
        path = os.path.expanduser(destination)
        os.makedirs(path, exist_ok=True)

        # download
        urllib.request.urlretrieve(url, path + fn)

        # unzip
        with zipfile.ZipFile(path + fn, 'r') as zip_ref:
            zip_ref.extractall(destination)

        # remove zipfile
        os.remove(destination + fn)


class SFUDataset(Dataset):
    def __init__(self, destination: str = 'images/SFU/'):
        self.destination = destination

    def load(self) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        print('===== Load dataset SFU dry--jan')

        # download images if necessary
        if not os.path.exists(self.destination):
            self.download(self.destination)

        # load images
        fns_db = sorted(glob(self.destination + 'dry/*.jpg'))
        fns_q = sorted(glob(self.destination + 'jan/*.jpg'))

        imgs_db = [np.array(Image.open(fn)) for fn in fns_db]
        imgs_q = [np.array(Image.open(fn)) for fn in fns_q]

        # create ground truth
        gt_data = np.load(self.destination + 'GT.npz')
        GThard = gt_data['GThard'].astype('bool')
        GTsoft = gt_data['GTsoft'].astype('bool')

        return imgs_db, imgs_q, GThard, GTsoft

    def download(self, destination: str):
        print('===== SFU dataset does not exist. Download to ' + destination + '...')

        fn = 'SFU.zip'
        url = 'https://www.tu-chemnitz.de/etit/proaut/datasets/' + fn

        # create folders
        path = os.path.expanduser(destination)
        os.makedirs(path, exist_ok=True)

        # download
        urllib.request.urlretrieve(url, path + fn)

        # unzip
        with zipfile.ZipFile(path + fn, 'r') as zip_ref:
            zip_ref.extractall(destination)

        # remove zipfile
        os.remove(destination + fn)

class TwoImgDataset(Dataset):
    def __init__(self, destination: str = 'images/MatchingPairs/'):
        self.destination = destination

    def load(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        print('===== Load dataset TwoImg')

        # download images if necessary
        if not os.path.exists(self.destination):
            self.download(self.destination)

        # load images
        fns_db = sorted(glob(self.destination + '1/*.jpg'))
        fns_q = sorted(glob(self.destination + '2/*.jpg'))

        imgs_db = [np.array(Image.open(fn)) for fn in fns_db]
        imgs_q = [np.array(Image.open(fn)) for fn in fns_q]

        # create ground truth
        # GThard = np.eye(len(imgs_db)).astype('bool')
        # GTsoft = convolve2d(GThard.astype('int'),
        #                     np.ones((17, 1), 'int'), mode='same').astype('bool')

        return imgs_db, imgs_q
        

    def download(self, destination: str):
        pass


class Tokyo247Dataset(Dataset):
    def __init__(self, destination: str = 'mini_VPR_datasets/Tokyo24_7/tokyo247_vpr_format/'):
        self.destination = destination

    def load(self) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        print('===== Load dataset Tokyo24/7')

        if not os.path.exists(self.destination):
            raise FileNotFoundError(f"Dataset directory not found at {self.destination}. Please run prepare_tokyo247.py first.")

        # Use i0 images as database (typically day) and i1 as queries (typically night)
        fns_db = sorted(glob(os.path.join(self.destination, 'p*/i0/*.jpg')))
        fns_q = sorted(glob(os.path.join(self.destination, 'p*/i1/*.jpg')))

        if not fns_db or not fns_q:
            raise FileNotFoundError(f"Could not find images in {self.destination}. Ensure the structure is p*/i*/*.jpg")
            
        if len(fns_db) != len(fns_q):
            print(f"Warning: Database and query sets have different numbers of images. DB: {len(fns_db)}, Q: {len(fns_q)}")

        imgs_db = [np.array(Image.open(fn)) for fn in fns_db]
        imgs_q = [np.array(Image.open(fn)) for fn in fns_q]

        # create ground truth assuming one-to-one correspondence
        num_places = min(len(imgs_db), len(imgs_q))
        GThard = np.eye(num_places).astype('bool')
        # Soft GT with a window of 1, adjust if needed
        GTsoft = convolve2d(GThard.astype('int'),
                            np.ones((3, 1), 'int'), mode='same').astype('bool')

        return imgs_db, imgs_q, GThard, GTsoft

    def download(self, destination: str):
        print("Tokyo24/7 dataset is not available for download via this script.")
        print("Please download it manually and use prepare_tokyo247.py to format it.")
        pass