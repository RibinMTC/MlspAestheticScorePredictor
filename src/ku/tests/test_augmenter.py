from __future__ import print_function
from __future__ import absolute_import
from src.ku import image_augmenter as aug, image_utils as iu
import numpy as np


def test_imageutils_exposes_augmenter():
    assert isinstance(iu.ImageAugmenter(np.ones(1)), aug.ImageAugmenter)

def test_cropout_and_crop():
    m = np.zeros((5,5,3))
    c = np.zeros((5,5,3))
    c[1:4,1:4,:] = 1
    
    # cropout
    assert np.array_equal(aug.cropout_patch(m, patch_size=(3,3), patch_position=(0.5,0.5), fill_val=1), c)
    assert np.array_equal(aug.ImageAugmenter(c).cropout((3,3), crop_pos=(0.5,0.5), fill_val=1).result, c)
    assert np.array_equal(aug.ImageAugmenter(c).cropout((3,3), crop_pos=(0.5,0.5), fill_val=0).result, m)

    # crop
    assert np.array_equal(aug.ImageAugmenter(c).crop((3,3), crop_pos=(0.5,0.5)).result, np.ones((3,3,3)))

def test_flip():
    m = np.zeros((5,5,3))
    ml, mr = [m]*2
    ml[0:2,0:2,:] = 1
    mr[0:2,-2:,:] = 1

    assert np.array_equal(aug.ImageAugmenter(m).fliplr().result, m)
    assert np.array_equal(aug.ImageAugmenter(ml).fliplr().result, mr)

def test_cropout_fills_image():
    ''' 
    Tests if by repeatedly cropping out patches, the whole image gets filled in with the `fill_val`=1.
    Tries to crop with different sizes.
    '''
    m = np.zeros((5,5,3))

    # fills with cropout_dim x cropout_dim
    for cropout_dim in range(1,5):
        a = aug.ImageAugmenter(m, remap=False)
        for i in range(1000):
            a.cropout((cropout_dim,cropout_dim), fill_val=1)
        assert np.array_equal(a.result, np.ones((5,5,3)))

def test_imshuffle():
    m = np.ones((4,4))
    assert np.array_equal(aug.imshuffle(m, [2,2]), np.ones((4,4)))

    m[:,0] = 0
    assert np.sum(aug.imshuffle(m, [4,4])==0)==4
    assert np.array_equal(aug.imshuffle(m, [1,1]), m)

    m = np.zeros((2,2))
    m[0,0] = 1
    for _ in range(1000):
        assert np.sum(aug.imshuffle_pair(m, m, [2,2]))<=2
        assert np.sum(aug.imshuffle_pair(m, 1-m, [2,2]))>=1
        
def test_imshuffle_pair_ratio():
    m1 = np.ones((4,4))
    m2 = np.zeros((4,4))

    for _ in range(1000):
        for ratio in [0,0.25,0.5,0.75,1]:
            assert np.sum(aug.imshuffle_pair(m1, m2, [4,4], ratio)) == ratio*16