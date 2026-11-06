from __future__ import division, print_function, absolute_import
import glob
import warnings
import os.path as osp
from .bases import BaseImageDataset
import re
import os

class UNIRNT(BaseImageDataset):
    dataset_dir = 'UNIRNT'

    def __init__(self, root='', verbose=True, **kwargs):
        super(UNIRNT, self).__init__()
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir)
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn(
                'The current data structure is deprecated.'
            )

        # self.train_dir = osp.join(self.data_dir, 'train') # 混合RGBNT201 RGBNT100 MSVR310的训练集
        
        self.train_dir = osp.join(self.data_dir, 'train3')# 混合RGBNT201 RGBNT100 的训练集
        
        self.query_dir_201 = '/data1/zhouzhongao/datasets/RGBNT201/test'
        self.gallery_dir_201 = '/data1/zhouzhongao/datasets/RGBNT201/test'
        
        self.query_dir_100 = '/data1/zhouzhongao/datasets/RGBNT100/rgbir/query'
        self.gallery_dir_100 = '/data1/zhouzhongao/datasets/RGBNT100/rgbir/bounding_box_test'
        
        self.query_dir_msvr = '/data1/zhouzhongao/datasets/MSVR310/query3'
        self.gallery_dir_msvr = '/data1/zhouzhongao/datasets/MSVR310/bounding_box_test'

        self._check_before_run()
        self.human_id =set()
        self.vehicle_id =set()
        # train = self._process_dir_train(self.train_dir, relabel=True)
        train = self._process_dir(self.train_dir, relabel=True)
        
        query_201 = self._process_dir_201(self.query_dir_201, relabel=False)
        gallery_201 = self._process_dir_201(self.gallery_dir_201, relabel=False)
        
        query_100 = self._process_dir_100(self.query_dir_100, relabel=False)
        gallery_100 = self._process_dir_100(self.gallery_dir_100, relabel=False)
        
        query_msvr = self._process_dir_msvr(self.query_dir_msvr, relabel=False)
        gallery_msvr = self._process_dir_msvr(self.gallery_dir_msvr, relabel=False)
        
        self.train = train
        
        self.query_201 = query_201
        self.gallery_201 = gallery_201
        
        self.query_100 = query_100
        self.gallery_100 = gallery_100
        
        self.query_msvr= query_msvr
        self.gallery_msvr= gallery_msvr
        
        if verbose:
            print("=> UNIRNT loaded")
            self.print_dataset_statistics(train, query_201, gallery_201)
            self.print_dataset_statistics(train, query_100, gallery_100)
            self.print_dataset_statistics(train, query_msvr, gallery_msvr)


        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        
        self.num_query_201_pids, self.num_query_201_imgs, self.num_query_201_cams, self.num_query_201_vids = self.get_imagedata_info(
            self.query_201)
        self.num_gallery_201_pids, self.num_gallery_201_imgs, self.num_gallery_201_cams, self.num_gallery_201_vids = self.get_imagedata_info(
            self.gallery_201)
        
        self.num_query_100_pids, self.num_query_100_imgs, self.num_query_100_cams, self.num_query_100_vids = self.get_imagedata_info(
            self.query_100)
        self.num_gallery_100_pids, self.num_gallery_100_imgs, self.num_gallery_100_cams, self.num_gallery_100_vids = self.get_imagedata_info(
            self.gallery_100)
        
        self.num_query_msvr_pids, self.num_query_msvr_imgs, self.num_query_msvr_cams, self.num_query_msvr_vids = self.get_imagedata_info(
            self.query_msvr)
        self.num_gallery_msvr_pids, self.num_gallery_msvr_imgs, self.num_gallery_msvr_cams, self.num_gallery_msvr_vids = self.get_imagedata_info(
            self.gallery_msvr)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir_201):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir_201):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
        if not osp.exists(self.query_dir_100):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir_100):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
        if not osp.exists(self.query_dir_msvr):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir_msvr):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
    def _process_dir(self, dir_path, relabel=False):
        img_paths_RGB = glob.glob(osp.join(dir_path, 'RGB', '*.jpg'))
        pid_container = set()
        for img_path_RGB in img_paths_RGB:
            jpg_name = img_path_RGB.split('/')[-1]
            # pid = int(jpg_name.split('_')[0][0:6])
            pid = int(jpg_name.split('_')[1][0:6])
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path_RGB in img_paths_RGB:
            img = []
            jpg_name = img_path_RGB.split('/')[-1]
            img_path_NI = osp.join(dir_path, 'NI', jpg_name)
            img_path_TI = osp.join(dir_path, 'TI', jpg_name)
            img.append(img_path_RGB)
            img.append(img_path_NI)
            img.append(img_path_TI)
            # pid = int(jpg_name.split('_')[0][0:6])
            pid = int(jpg_name.split('_')[1][0:6])
            camid = 0
            # trackid = -1
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            if jpg_name.split('_')[0] =='h': #来自rgbnt201
                trackid = 0
                self.human_id.add(pid)
            elif jpg_name.split('_')[0] =='v':#来自rgbnt100
                trackid = 1
                self.vehicle_id.add(pid)
            data.append((img, pid, camid, trackid)) # trackid来代表类别
            # print("11111")
        return data
    # def _process_dir_train(self, dir_path, relabel=False):
    #     img_paths_RGB = glob.glob(osp.join(dir_path, 'RGB', '*.jpg'))
    #     pid_container = set()
    #     for img_path_RGB in img_paths_RGB:
    #         jpg_name = img_path_RGB.split('/')[-1]
    #         category = jpg_name.split('_')[0]
    #         pid = int(jpg_name.split('_')[1][0:6])
    #         pid = category + str(pid)
    #         pid_container.add(pid)
    #     pid2label = {pid: label for label, pid in enumerate(pid_container)}

    #     data = []
    #     for img_path_RGB in img_paths_RGB:
    #         img = []
    #         jpg_name = img_path_RGB.split('/')[-1]
    #         img_path_NI = osp.join(dir_path, 'NI', jpg_name)
    #         img_path_TI = osp.join(dir_path, 'TI', jpg_name)
    #         img.append(img_path_RGB)
    #         img.append(img_path_NI)
    #         img.append(img_path_TI)
    #         category = jpg_name.split('_')[0]
    #         pid = int(jpg_name.split('_')[1][0:6])
    #         pid = category + str(pid)
    #         camid = 0
    #         if category =='h': #来自rgbnt201
    #             category_id = 0
    #         elif category =='v':#来自rgbnt100
    #             category_id = 1
    #         elif category =='m':#来自msvr
    #             category_id = 2
    #         camid -= 1  # index starts from 0
    #         if relabel:
    #             pid = pid2label[pid]
    #         data.append((img, pid, camid,category_id)) # 删去了trackid
    #         # print("11111")
    #     return data
    
    def _process_dir_201(self, dir_path, relabel=False):
        img_paths_RGB = glob.glob(osp.join(dir_path, 'RGB', '*.jpg'))
        pid_container = set()
        for img_path_RGB in img_paths_RGB:
            jpg_name = img_path_RGB.split('/')[-1]
            pid = int(jpg_name.split('_')[0][0:6])
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path_RGB in img_paths_RGB:
            img = []
            jpg_name = img_path_RGB.split('/')[-1]
            img_path_NI = osp.join(dir_path, 'NI', jpg_name)
            img_path_TI = osp.join(dir_path, 'TI', jpg_name)
            img.append(img_path_RGB)
            img.append(img_path_NI)
            img.append(img_path_TI)
            pid = int(jpg_name.split('_')[0][0:6])
            camid = int(jpg_name.split('_')[1][3])
            trackid = -1
            camid -= 1  # index starts from 0
            species_id = 0
            if relabel:
                pid = pid2label[pid]
            data.append((img, pid, camid, species_id))
            # print("11111")
        return data
    
    def _process_dir_100(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            #pdb.set_trace()
            #if pid == -1: continue  # junk images are just ignored
            assert 1 <= pid <= 600  # pid == 0 means background
            assert 1 <= camid <= 8
            trackid = -1
            camid -= 1  # index starts from 0
            species_id = 1
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid,species_id))
        return dataset
    
    def _process_dir_msvr(self, dir_path, relabel=False):
        vid_container = set()
        for vid in os.listdir(dir_path):
            vid_container.add(int(vid))
        vid2label = {vid: label for label, vid in enumerate(vid_container)}

        dataset = []
        for vid in os.listdir(dir_path):
            vid_path = osp.join(dir_path, vid)
            r_data = os.listdir(osp.join(vid_path, 'vis'))
            for img in r_data:
                r_img_path = osp.join(vid_path, 'vis', img)
                n_img_path = osp.join(vid_path, 'ni', img)
                t_img_path = osp.join(vid_path, 'th', img)
                vid = int(img[0:4])
                camid = int(img[11])
                sceneid = int(img[6:9])  # scene id
                # species_id = 2
                assert 0 <= camid <= 7
                if relabel:
                    vid = vid2label[vid]
                dataset.append(((r_img_path, n_img_path, t_img_path), vid, camid,sceneid))
        return dataset