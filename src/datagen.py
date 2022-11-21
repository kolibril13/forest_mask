import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# Data generator for the model
# Tiles a xr.Dataset into multiple tiles and returns them in batches (row-wise)
# Also splits the data into training/validation/test
class CustomImageDataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, ds, tilesize, sampletype):
        self.ds   = ds
        self.tilesize = tilesize
        self.ylen = self.ds.y.size // self.tilesize
        self.xlen = self.ds.x.size // self.tilesize
        self.sampletype = sampletype
        
    def __len__(self):
        
        return self.ylen

    def __getitem__(self, index):
        
        red       = self.ds.Band1[index*self.tilesize:(index+1)*self.tilesize,:-(self.ds.x.size%self.tilesize)]
        green     = self.ds.Band2[index*self.tilesize:(index+1)*self.tilesize,:-(self.ds.x.size%self.tilesize)]
        blue      = self.ds.Band3[index*self.tilesize:(index+1)*self.tilesize,:-(self.ds.x.size%self.tilesize)]
        forest    = self.ds.forest_mask[index*self.tilesize:(index+1)*self.tilesize,:-(self.ds.x.size%self.tilesize)]
        
        rgb       = np.array([red,green,blue]).transpose(1,2,0)
        forest    = np.array(forest)
        
        rgb_tiles    = np.array(np.split(rgb, self.xlen,axis=1))
        target_tiles = np.array(np.split(forest, self.xlen,axis=1))
        
        # Depending on sampletype, return training, validation or test set (complete set)
        if self.sampletype == 'training' or self.sampletype == 'validation':
            rgb_tiles_tr, rgb_tiles_val, target_tiles_tr, target_tiles_val = train_test_split(rgb_tiles, target_tiles, shuffle=True, test_size=0.1, random_state=0)
            
            if self.sampletype == 'training': return rgb_tiles_tr, target_tiles_tr
            else:                             return rgb_tiles_val, target_tiles_val
        
        if self.sampletype == 'test': return rgb_tiles, target_tiles
        
        return None