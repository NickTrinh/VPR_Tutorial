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
import imageio.v2 as imageio


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
        self.db_paths = [] # Add attribute to store db paths
        self.q_paths = [] # Add attribute to store q paths

    def load(self) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        print('===== Load dataset GardensPoint day_right--night_right')

        # download images if necessary
        if not os.path.exists(self.destination):
            self.download(self.destination)

        # get image paths
        path_db = os.path.join(self.destination, 'day_right')
        path_q = os.path.join(self.destination, 'night_right')
        
        self.db_paths = sorted(glob(os.path.join(path_db, '*.jpg')))
        self.q_paths = sorted(glob(os.path.join(path_q, '*.jpg')))

        # load images
        imgs_db = [imageio.imread(p) for p in self.db_paths]
        imgs_q = [imageio.imread(p) for p in self.q_paths]

        # ground truth
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
        self.fns_db_path = '100909_0845' # Add path for db images
        self.fns_q_path = '180809_1545' # Add path for query images

    def load(self) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        print('===== Load dataset StLucia 100909_0845--180809_1545 (small version)')

        # download images if necessary
        if not os.path.exists(self.destination):
            self.download(self.destination)

        # load images
        fns_db = sorted(glob(self.destination + self.fns_db_path + '/*.jpg'))
        fns_q = sorted(glob(self.destination + self.fns_q_path + '/*.jpg'))

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
        self.fns_db_path = 'dry' # Add path for db images
        self.fns_q_path = 'jan' # Add path for query images

    def load(self) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        print('===== Load dataset SFU dry--jan')

        # download images if necessary
        if not os.path.exists(self.destination):
            self.download(self.destination)

        # load images
        fns_db = sorted(glob(self.destination + self.fns_db_path + '/*.jpg'))
        fns_q = sorted(glob(self.destination + self.fns_q_path + '/*.jpg'))

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


class GardensPointLandmarkDataset(Dataset):
    def __init__(self, destination: str = 'images/GardensPoint_Landmark/'):
        self.destination = destination

    def load(self) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        print('===== Load dataset GardensPoint_Landmark (day_left vs night_right)')
        
        db_paths = sorted(glob(os.path.join(self.destination, 'p*', '*_day_left_*.jpg')))
        q_paths = sorted(glob(os.path.join(self.destination, 'p*', '*_night_right_*.jpg')))

        imgs_db = [imageio.imread(p) for p in db_paths]
        imgs_q = [imageio.imread(p) for p in q_paths]

        # Get the place ID from the path (e.g., 'images/GardensPoint_Landmark/p10')
        db_place_ids = [int(os.path.basename(os.path.dirname(p))[1:]) for p in db_paths]
        q_place_ids = [int(os.path.basename(os.path.dirname(p))[1:]) for p in q_paths]

        num_db = len(imgs_db)
        num_q = len(imgs_q)
        
        GThard = np.zeros((num_db, num_q), dtype=bool)
        for i in range(num_db):
            for j in range(num_q):
                if db_place_ids[i] == q_place_ids[j]:
                    GThard[i, j] = True

        # No soft ground truth for this dataset
        GTsoft = np.zeros_like(GThard, dtype=bool)

        return imgs_db, imgs_q, GThard, GTsoft


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


class PlaceConditionsDataset(Dataset):
    def __init__(self, destination: str, db_condition: str, q_condition: str):
        self.destination = destination
        self.db_condition = db_condition
        self.q_condition = q_condition
        self.db_paths: List[str] = []
        self.q_paths: List[str] = []

    def load(self) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        print(f"===== Load dataset (Place#### across conditions: {self.db_condition} -> {self.q_condition})")

        db_dir = os.path.join(self.destination, self.db_condition)
        q_dir = os.path.join(self.destination, self.q_condition)
        if not (os.path.isdir(db_dir) and os.path.isdir(q_dir)):
            raise FileNotFoundError(f"Expected condition folders at {db_dir} and {q_dir}")

        # Reset collected paths on each load
        self.db_paths = []
        self.q_paths = []

        def list_place_files(folder: str) -> List[str]:
            files = sorted(glob(os.path.join(folder, 'Place*_Cond*_G*.jpg')))
            files += sorted(glob(os.path.join(folder, 'Place*_Cond*_G*.jpeg')))
            files += sorted(glob(os.path.join(folder, 'Place*_Cond*_G*.png')))
            return files

        db_files = list_place_files(db_dir)
        q_files = list_place_files(q_dir)
        if not db_files or not q_files:
            raise FileNotFoundError(f"No place files found under {db_dir} or {q_dir}")

        def parse_place_gid(path: str) -> Tuple[int, int]:
            base = os.path.basename(path)
            # Expect Place####_CondCC_GGG.ext
            try:
                place_part, cond_part, g_part_ext = base.split('_')
                pid = int(place_part.replace('Place', ''))
                gid = int(g_part_ext.split('.')[0].replace('G', ''))
                return pid, gid
            except Exception:
                return -1, -1

        def index_by_place(files: List[str]):
            idx = {}
            for f in files:
                pid, gid = parse_place_gid(f)
                if pid < 0 or gid < 0:
                    continue
                if pid not in idx:
                    idx[pid] = {}
                idx[pid][gid] = f
            return idx

        db_idx = index_by_place(db_files)
        q_idx = index_by_place(q_files)

        common_pids = sorted(set(db_idx.keys()) & set(q_idx.keys()))

        imgs_db: List[np.ndarray] = []
        imgs_q: List[np.ndarray] = []

        for pid in common_pids:
            common_g = sorted(set(db_idx[pid].keys()) & set(q_idx[pid].keys()))
            if not common_g:
                continue
            # Pair all available group IDs for this place (align gid k with gid k)
            for gid in common_g:
                db_path = db_idx[pid][gid]
                q_path = q_idx[pid][gid]
                self.db_paths.append(db_path)
                self.q_paths.append(q_path)
                imgs_db.append(np.array(Image.open(db_path)))
                imgs_q.append(np.array(Image.open(q_path)))

        num = min(len(imgs_db), len(imgs_q))
        imgs_db = imgs_db[:num]
        imgs_q = imgs_q[:num]

        GThard = np.eye(num).astype('bool')
        GTsoft = convolve2d(GThard.astype('int'), np.ones((3, 1), 'int'), mode='same').astype('bool')

        return imgs_db, imgs_q, GThard, GTsoft

    def download(self, destination: str):
        # Prepared local datasets only; nothing to download here.
        return