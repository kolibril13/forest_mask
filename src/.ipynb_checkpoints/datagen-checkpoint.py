import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# Data generator for the model
# Tiles a xr.Dataset into multiple tiles and returns them in batches (row-wise)
# Also splits the data into training/validation/test
class CustomImageDataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, ds, tilesize, sampletype, data_augmentation):
        
        self.ds   = ds
        self.tilesize = tilesize
        self.ylen = self.ds.y.size // self.tilesize
        self.xlen = self.ds.x.size // self.tilesize
        self.sampletype = sampletype
        self.data_augmentation = data_augmentation
        
    def __len__(self):
        
        return self.ylen

    def __getitem__(self, index):
        
        red       = self.ds.red[index*self.tilesize:(index+1)*self.tilesize,:-(self.ds.x.size%self.tilesize)]
        green     = self.ds.green[index*self.tilesize:(index+1)*self.tilesize,:-(self.ds.x.size%self.tilesize)]
        blue      = self.ds.blue[index*self.tilesize:(index+1)*self.tilesize,:-(self.ds.x.size%self.tilesize)]
        forest    = self.ds.forest_mask[index*self.tilesize:(index+1)*self.tilesize,:-(self.ds.x.size%self.tilesize)]
        
        rgb       = np.array([red,green,blue]).transpose(1,2,0)
        forest    = np.array(forest)
        
        rgb_tiles    = np.array(np.split(rgb, self.xlen, axis=1))
        target_tiles = np.array(np.split(forest, self.xlen, axis=1))
        
        # Depending on sampletype, return training, validation or test set (complete set)
        # Data augmentation can be included if data_augmentation is set to True 
        # (rotating the images by 90, 180, and 270 degrees)
        if self.sampletype == 'training' or self.sampletype == 'validation':
            
            rgb_tiles_tr, rgb_tiles_val, target_tiles_tr, target_tiles_val = train_test_split(rgb_tiles, target_tiles, shuffle=True, test_size=0.1, random_state=0)
            
            if self.sampletype == 'training' and self.data_augmentation == True:
                
                rgb_tiles_tr    = np.concatenate((rgb_tiles_tr,
                                               tf.image.rot90(image=rgb_tiles_tr),
                                               tf.image.rot90(image=rgb_tiles_tr, k=2),
                                               tf.image.rot90(image=rgb_tiles_tr, k=3)),
                                              axis=0)
                
                target_tiles_tr = np.concatenate((target_tiles_tr,
                                                  tf.image.rot90(np.expand_dims(target_tiles_tr, axis=-1))[:,:,:,0],
                                                  tf.image.rot90(np.expand_dims(target_tiles_tr, axis=-1), k=2)[:,:,:,0],
                                                  tf.image.rot90(np.expand_dims(target_tiles_tr, axis=-1), k=3)[:,:,:,0]),
                                                 axis=0)
                
                ng              = np.random.RandomState(11)
                indexes         = ng.permutation(rgb_tiles_tr.shape[0])
                rgb_tiles_tr    = rgb_tiles_tr[indexes]
                target_tiles_tr = target_tiles_tr[indexes]
                
                return rgb_tiles_tr, target_tiles_tr
            
            elif self.sampletype == 'training' and self.data_augmentation == False:
                
                return rgb_tiles_tr, target_tiles_tr
            
            else:
                
                return rgb_tiles_val, target_tiles_val
        
        if self.sampletype == 'test':
            
            return rgb_tiles, target_tiles
        
        return None