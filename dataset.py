## Author: Lishuo Pan 2020/4/18

import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        # TODO: load dataset, make mask list
        self.img    = h5py.File(path[0],'r')['data']
        self.mask   = h5py.File(path[1],'r')['data']
        self.labels = np.load(path[2], allow_pickle=True)
        self.bbox   = np.load(path[3], allow_pickle=True)
           
        #Transforms
        self.normalize = transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
        
        self.aligned_masks = [] #List of numpy array
        
        i = 0 
        mask_shape = self.mask[0].shape
        for l in range(self.labels.shape[0]):
            length = self.labels[l].size
            clubbed_mask = []
            for idx in range(length):
                clubbed_mask.append(self.mask[i,:,:])
                i+=1
                
            self.aligned_masks.append(clubbed_mask)
                
    # output:
        # transed_img
        # label
        # transed_mask
        # transed_bbox
    def __getitem__(self, index):
        # TODO: __getitem__
        img   = self.img[index,:,:,:]
        label = self.labels[index]
        mask  = self.aligned_masks[index]
        bbox  = self.bbox[index]
        
        label = torch.tensor(label, dtype=torch.float)
        transed_img, transed_mask, transed_bbox = self.pre_process_batch(img, mask, bbox)
        
        # check flag
        assert transed_img.shape == (3, 800, 1088)
        assert transed_bbox.shape[0] == transed_mask.shape[0]
        return transed_img, label, transed_mask, transed_bbox
    
    def __len__(self):
        return self.img.shape[0]

    # This function take care of the pre-process of img,mask,bbox
    # in the input mini-batch
    # input:
        # img: 3*300*400
        # mask: 3*300*400
        # bbox: n_box*4
    def pre_process_batch(self, img, mask, bbox):
        # TODO: image preprocess
        scale_factor_x = 800/300
        scale_factor_y = 1066/400
        
        transed_img   = torch.tensor(img.astype(np.float), dtype=torch.float)
        transed_bbox  = torch.tensor(bbox, dtype=torch.float)
            
        transed_img =  transed_img/255.0 #Normalization
        transed_img =  torch.unsqueeze(transed_img, 0)
        transed_img  = torch.nn.functional.interpolate(transed_img, size=(800, 1066), mode='bilinear') #Interpolation
        transed_img =  self.normalize(transed_img[0])
        #print(transed_img.shape)
        transed_img =  torch.nn.functional.pad(transed_img, pad=(11,11), mode='constant',value=0)

        # transed_masks = torch.zeros((len(mask),3,800,1088))
        transed_masks = torch.zeros((len(mask),3,800,1088))
        for i, m in enumerate(mask):
            transed_mask = torch.tensor(m.astype(np.uint8), dtype=torch.float)
            transed_mask = torch.unsqueeze(transed_mask, 0)
            transed_mask = torch.cat(3*[transed_mask])
            transed_mask = torch.unsqueeze(transed_mask, 0)
            transed_mask = torch.nn.functional.interpolate(transed_mask,size=(800, 1066), mode='bilinear')
            transed_mask = torch.nn.functional.pad(transed_mask, pad=(11,11), mode='constant',value=0)
            transed_mask[transed_mask > 0.5] = 1
            transed_mask[transed_mask < 0.5] = 0
            transed_masks[i] = transed_mask[0,0,:,:]

        transed_bbox[:,0] = transed_bbox[:,0] * scale_factor_x
        transed_bbox[:,2] = transed_bbox[:,2] * scale_factor_x
        transed_bbox[:,1] = transed_bbox[:,1] * scale_factor_y
        transed_bbox[:,3] = transed_bbox[:,3] * scale_factor_y
        transed_bbox[:,0] += 11 
        transed_bbox[:,2] += 11 #Accounting for changes in x due to padding
    
        # check flag
        assert transed_img.shape == (3, 800, 1088)
        assert transed_bbox.shape[0] == transed_masks.shape[0]
        return transed_img, transed_masks, transed_bbox


class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    # output:
        # img: (bz, 3, 800, 1088)
        # label_list: list, len:bz, each (n_obj,)
        # transed_mask_list: list, len:bz, each (n_obj, 800,1088)
        # transed_bbox_list: list, len:bz, each (n_obj, 4)
        # img: (bz, 3, 300, 400)
    def collect_fn(self, batch):
        bz = len(batch)
        transed_img_list = []
        label_list = []
        transed_mask_list = []
        transed_bbox_list = []

        for transed_img, label, transed_mask, transed_bbox in batch:
          transed_img_list.append(transed_img)
          label_list.append(label)
          transed_mask_list.append(transed_mask)
          transed_bbox_list.append(transed_bbox)

        return torch.stack(transed_img_list, dim = 0), \
                label_list,\
                transed_mask_list,\
                transed_bbox_list

        # TODO: collect_fn

    def loader(self):
        
        return torch.utils.data.DataLoader(self.dataset, 
                                    batch_size= self.batch_size, 
                                    shuffle= self.shuffle, 
                                    sampler=None, 
                                    batch_sampler=None, 
                                    num_workers= self.num_workers, 
                                    collate_fn= self.collect_fn)

## Visualize debugging
if __name__ == '__main__':
    # file path and make a list
    imgs_path =   './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path =  './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = './data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = './data/hw3_mycocodata_bboxes_comp_zlib.npy'
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)
    ###Debug###

    ## Visualize debugging
    # --------------------------------------------
    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset
    # set seed
    torch.random.manual_seed(0)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # push the randomized training data into the dataloader

    batch_size = 2
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # reset Shuffle to True
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()

    mask_color_list = ["jet", "ocean", "Spectral", "spring", "cool"]

    # loop the image
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for iter, data in enumerate(train_loader, 0):

        img, label, mask, bbox = [data[i] for i in range(len(data))]
        # check flag
        assert img.shape == (batch_size, 3, 800, 1088)
        assert len(mask) == batch_size

        label = [label_img.to(device) for label_img in label]
        mask = [mask_img.to(device) for mask_img in mask]
        bbox = [bbox_img.to(device) for bbox_img in bbox]

        # plot the origin img
        invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

        for i in range(batch_size):
            ## TODO: plot images with annotations      
            combined_mask = np.zeros((img.shape[2],img.shape[3],img.shape[1]))

            fig = plt.figure()
            ax = fig.add_subplot(111)

            for j, l in enumerate(label[i]):
              xA, xB, yA, yB = bbox[i][j,0], bbox[i][j,2], bbox[i][j,1], bbox[i][j,3]
              if l == 1:
                   combined_mask[:,:,0] += (mask[i][j,0,:,:] * 255).cpu().numpy()
                   rect = patches.Rectangle((xA,yA), xB-xA, yB-yA, linewidth=1, edgecolor='r',facecolor='none')
              if l == 2:
                  combined_mask[:,:,1] += (mask[i][j,1,:,:] * 255).cpu().numpy()
                  rect = patches.Rectangle((xA,yA), xB-xA, yB-yA, linewidth=1, edgecolor='g',facecolor='none')
              if l == 3: 
                  combined_mask[:,:,2] += (mask[i][j,2,:,:] * 255).cpu().numpy()
                  rect = patches.Rectangle((xA,yA), xB-xA, yB-yA, linewidth=1, edgecolor='b',facecolor='none')
              ax.add_patch(rect)

            origin_img = invTrans(img[i,:,:,:]).numpy().transpose(1,2,0)
            mask_to_plot = combined_mask + (origin_img*(1 - (combined_mask/255))*255)
            ax.imshow(origin_img)
            ax.imshow(np.uint8(mask_to_plot),alpha = 0.5)
            # plt.savefig("./testfig/visualtrainset"+str(iter)+".png")
            plt.show()

        if iter == 10:
          break
