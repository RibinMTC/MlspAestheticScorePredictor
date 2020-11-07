from __future__ import print_function
from __future__ import absolute_import

from src.ku import image_augmenter as aug, image_utils as iu
from src.ku import generators as gr

import shutil, os

def test_basic_resize_check_save_h5():
    
    # resize
    iu.resize_folder('images/', 'images_temp/',
                     image_size_dst=(50,50), overwrite=True)
    image_list = iu.glob_images('images_temp', verbose=False)
    assert image_list
    ims = iu.read_image_batch(image_list)
    assert ims.shape == (4, 50, 50, 3)

    # check
    failed_images, all_images = iu.check_images('images_temp/')
    assert len(failed_images)==0
    assert len(all_images)==4

    # save to h5
    iu.save_images_to_h5('images_temp', 'images.h5', 
                         overwrite=True)
    with gr.H5Helper('images.h5') as h:
        assert list(h.hf.keys()) == sorted(all_images)
        
    # clean-up
    shutil.rmtree('images_temp')
    os.unlink('images.h5')
    
    
def test_augment_folder():
    
    path_src= 'images/'
    path_dst='images_aug/'

    def process_gen():
        for num_patch in [(i,j) for i in [1,2,4,8] for j in [1,2,4,8]]:
            fn = lambda im, **kwargs: aug.imshuffle(im, num_patch)
            yield fn, dict(num_patch=num_patch)

    ids_aug = iu.augment_folder(path_src, path_dst, 
                                process_gen, verbose=True)

    assert len(ids_aug)==64

    (image_path, ext) = os.path.split(ids_aug.iloc[0,:].image_path)
    _, file_names = iu.glob_images('{}{}/'.format(path_dst,image_path), split=True)

    first_group_names = list(ids_aug.groupby('num_patch'))[0][1].image_name
    assert sorted(first_group_names) == sorted(file_names)

    shutil.rmtree(path_dst)