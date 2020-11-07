from __future__ import print_function
from __future__ import absolute_import
from src.ku import generators as gr
from src.ku import generic as gen, image_utils as iu
from munch import Munch
import pandas as pd, numpy as np
import pytest, shutil

gen_params = Munch(batch_size    = 2,
                   data_path     ='images',
                   input_shape   = (224,224,3),
                   inputs        = ['filename'],
                   outputs       = ['score'],
                   shuffle       = False,
                   fixed_batches = True)

ids = pd.read_csv('ids.csv', encoding='latin-1')

def test_correct_df_input():
    assert (np.all(ids.columns == ['filename', 'score']))
    assert (np.all(ids.score == range(1,5)))

def test_init_DataGeneratorDisk():          
    g = gr.DataGeneratorDisk(ids, **gen_params)
    assert isinstance(g[0], tuple)
    assert isinstance(g[0][0], list)
    assert isinstance(g[0][1], list)
    assert (gen.get_sizes(g[0]) == '([array<2,224,224,3>], [array<2,1>])')
    assert (np.all(g[0][1][0] == np.array([[1],[2]])))
    
def test_read_fn_DataGeneratorDisk():
    import os
    def read_fn(name, g):
        # g is the parent generator object
        # name is the image name read from the DataFrame
        image_path = os.path.join(g.data_path, name)
        return iu.resize_image(iu.read_image(image_path), (100,100))        
        
    g = gr.DataGeneratorDisk(ids, read_fn=read_fn, **gen_params)
    gen.get_sizes(g[0]) =='([array<2,100,100,3>], [array<2,1>])'

def test_process_args_DataGeneratorDisk():
    def preproc(im, arg):
        return np.zeros(1) + arg

    gen_params_local = gen_params.copy()
    gen_params_local.process_fn   = preproc
    gen_params_local.process_args = {'filename': 'filename_args'}
    gen_params_local.batch_size   = 4

    ids_local = ids.copy()
    ids_local['filename_args'] = range(len(ids_local))

    g = gr.DataGeneratorDisk(ids_local, **gen_params_local)
    x = g[0][0]
    assert np.array_equal(np.squeeze(x[0].T), np.arange(gen_params_local.batch_size))

    
def test_get_sizes():
    x = np.array([[1,2,3]])
    assert gen.get_sizes(([x.T],1,[4,5])) == '([array<3,1>], <1>, [<1>, <1>])'
    assert gen.get_sizes(np.array([[1,[1,2]]])) == 'array<1,2>'

def test_DataGeneratorDisk():        
    g = gr.DataGeneratorDisk(ids, **gen_params)
    
    g.inputs = ['filename', 'filename']
    assert gen.get_sizes(g[0]) == '([array<2,224,224,3>, array<2,224,224,3>], [array<2,1>])'

    g.inputs_df = ['score', 'score']
    g.inputs = []
    g.outputs = []
    assert gen.get_sizes(g[0]) == '([array<2,2>], [])'

    g.inputs_df = [['score'], ['score','score']]
    assert gen.get_sizes(g[0]) == '([array<2,1>, array<2,2>], [])'

    g.inputs_df = []
    g.outputs = ['score']
    assert gen.get_sizes(g[0]) == '([], [array<2,1>])'

    g.outputs = ['score',['score']]
    with pytest.raises(AssertionError): g[0]

    g.outputs = [['score'],['score']]
    assert gen.get_sizes(g[0]) == '([], [array<2,1>, array<2,1>])'

def test_H5Reader_and_Writer():
    with gen.H5Helper('data.h5', overwrite=True) as h:
        data = np.expand_dims(np.array(ids.score), 1)
        h.write_data(data, list(ids.filename))

    with gen.H5Helper('data.h5', 'r') as h:
        data = h.read_data(list(ids.filename))
        assert all(data == np.array([[1],[2],[3],[4]]))
        
def test_DataGeneratorHDF5():
    gen_params_local = gen_params.copy()
    gen_params_local.update(data_path='data.h5', inputs=['filename'])    
    g = gr.DataGeneratorHDF5(ids, **gen_params_local)
    
    assert gen.get_sizes(g[0]) == '([array<2,1>], [array<2,1>])'
    
    g.inputs_df = ['score', 'score']
    g.inputs = []
    g.outputs = []
    assert gen.get_sizes(g[0]) == '([array<2,2>], [])'

    g.inputs_df = [['score'], ['score','score']]
    assert gen.get_sizes(g[0]) == '([array<2,1>, array<2,2>], [])'

    g.inputs_df = []
    g.outputs = ['score']
    assert gen.get_sizes(g[0]) == '([], [array<2,1>])'

    g.outputs = ['score',['score']]
    with pytest.raises(AssertionError): g[0]

    g.outputs = [['score'],['score']]
    assert gen.get_sizes(g[0]) == '([], [array<2,1>, array<2,1>])'
    
def test_process_args_DataGeneratorHDF5():
    def preproc(im, *arg):
        if arg:
            return np.zeros(im.shape) + arg
        else:
            return im

    gen_params_local = gen_params.copy()
    gen_params_local.update(process_fn = preproc,
                            data_path = 'data.h5', 
                            inputs    = ['filename', 'filename1'],
                            process_args = {'filename' :'args'},
                            batch_size = 4,
                            shuffle    = False)

    ids_local = ids.copy()
    ids_local['filename1'] = ids_local['filename']
    ids_local['args'] = range(len(ids_local))
    ids_local['args1'] = range(len(ids_local),0,-1)

    g = gr.DataGeneratorHDF5(ids_local, **gen_params_local)

    assert np.array_equal(np.squeeze(g[0][0][0]), np.arange(4))
    assert np.array_equal(np.squeeze(g[0][0][1]), np.arange(1,5))
    assert np.array_equal(np.squeeze(g[0][1]), np.arange(1,5))
    
def test_multi_process_args_DataGeneratorHDF5():
    def preproc(im, arg1, arg2):
        return np.zeros(1) + arg1 + arg2

    gen_params_local = gen_params.copy()
    gen_params_local.process_fn = preproc
    gen_params_local.process_args  = {'filename': ['filename_args','filename_args']}
    gen_params_local.batch_size = 4

    ids_local = ids.copy()
    ids_local['filename_args'] = range(len(ids_local))

    g = gr.DataGeneratorDisk(ids_local, **gen_params_local)
    x = g[0]
    assert np.array_equal(np.squeeze(x[0][0].T), np.arange(4)*2)
    
def test_callable_outputs_DataGeneratorHDF5():
    d = {'features': [1, 2, 3, 4, 5],
         'mask': [1, 0, 1, 1, 0]}
    df = pd.DataFrame(data=d)

    def filter_features(df):
        return np.array(df.loc[df['mask']==1,['features']])

    gen_params_local = gen_params.copy()
    gen_params_local.update(data_path = None, 
                            outputs   = filter_features,
                            inputs    = [],
                            inputs_df = ['features'],
                            shuffle   = False,
                            batch_size= 5)

    g = gr.DataGeneratorHDF5(df, **gen_params_local)
    assert gen.get_sizes(g[0]) == '([array<5,1>], array<3,1>)'
    assert all(np.squeeze(g[0][0]) == np.arange(1,6))
    assert all(np.squeeze(g[0][1]) == [1,3,4])
    
def test_multi_return_proc_fn_DataGeneratorDisk():
    gen_params_local = gen_params.copy()
    gen_params_local.process_fn = lambda im: [im, im+1]
    g = gr.DataGeneratorDisk(ids.copy(), **gen_params_local)
        
    assert np.array_equal(g[0][0][0], g[0][0][1]-1)
    assert np.array_equal(g[0][1][0], np.array([[1],[2]]))
    
def test_multi_return_and_read_fn_DataGeneratorDisk():    
    def read_fn(*args):
        g = args[1]
        score = np.float32(g.ids[g.ids.filename==args[0]].score)
        return np.ones((3,3)) * score

    gen_params_local = gen_params.copy()
    gen_params_local.batch_size = 3
    gen_params_local.read_fn = read_fn
    gen_params_local.process_fn = lambda im: [im+1, im+2]

    g = gr.DataGeneratorDisk(ids, **gen_params_local)
    assert np.array_equal(g[0][0][0], g[0][0][1]-1)
    assert np.array_equal(g[0][0][1][0,...], np.ones((3,3))*3.)
    
def test_generator_len_with_group_by_DataGeneratorDisk():
    size = 10
    ids_defa = pd.read_csv(u'ids.csv', encoding='latin-1')
    fnames = np.concatenate([ids_defa.filename.values]*3)[:size]
    ids  = pd.DataFrame(dict(cats  = ['cat{}'.format(i) for i in range(size)],
                            dogs  = ['dog{}'.format(i) for i in range(size)],
                            image_name = fnames,
                            group = [i//4 for i in range(10)]))

    gen_params = Munch(batch_size    = 1,
                       inputs        = ['image_name'],
                       outputs       = ['dogs'],
                       data_path     ='images',
                       group_by      = 'group',
                       shuffle       = False,
                       fixed_batches = True)

    for batch_size, len_g in zip(range(1, 5), [10, 5, 5, 3]):
        gen_params.batch_size = batch_size
        g = gr.DataGeneratorDisk(ids, **gen_params)
        assert len(g)==len_g
        a = g.ids_index.groupby('batch_index').group_by.mean().values
        b = g.ids_index.groupby('batch_index').group_by.last().values
        assert np.array_equal(a, b)

    gen_params.group_by = None
    for batch_size, len_g in zip(range(1, 5), [10, 5, 3, 2]):
        gen_params.batch_size = batch_size
        g = gr.DataGeneratorDisk(ids, **gen_params)
        assert len(g)==len_g

    gen_params.fixed_batches = False
    for batch_size, len_g in zip(range(1, 5), [10, 5, 4, 3]):
        gen_params.batch_size = batch_size
        g = gr.DataGeneratorDisk(ids, **gen_params)
        assert len(g)==len_g

def test_group_names_DataGeneratorDisk():
    
    iu.resize_folder('images/', 'images1/', image_size_dst=(100,100), overwrite=True)

    gp = gen_params.copy()
    gp.inputs = ['filename']
    gp.group_names = ['images/']
    gp.data_path   = ''
    g = gr.DataGeneratorDisk(ids, **gp)
    assert gen.get_sizes(g[0]) == '([array<2,224,224,3>], [array<2,1>])'

    gp.group_names = ['images/', 'images1/']
    g = gr.DataGeneratorDisk(ids, **gp)
    assert gen.get_sizes(g[0]) == '([array<2,224,224,3>, array<2,100,100,3>], [array<2,1>])'

    gp.group_names = [['images/'], ['images1/']]
    sizes = []
    for i in range(100):
        g = gr.DataGeneratorDisk(ids, **gp)
        sizes.append(g[0][0][0].shape[1])

    assert np.unique(sizes).shape[0]>1

    shutil.rmtree('images1/')
    
def test_random_group_DataGeneratorDisk():
  
    iu.resize_folder('images/', 'base/images100/', image_size_dst=(100,100), overwrite=True)
    iu.resize_folder('images/', 'base/images50/', image_size_dst=(50,50), overwrite=True)

    gp = gen_params.copy()
    gp.inputs       = ['filename']
    gp.data_path    = ''
    gp.group_names  = ['base']
    gp.random_group = True
    g = gr.DataGeneratorDisk(ids, **gp)
    
    assert np.array_equal(np.unique([x[0][0].shape[1] 
                              for i in range(100) for x in g]), [50,100])
    
    shutil.rmtree('base/')
    
def test_basics_deterministic_shuffle_consistency_group_by():

    ids = pd.DataFrame(dict(a = range(10), 
                            b = list(range(9,-1,-1)),
                            c = np.arange(10)<5))
    
    gen_params = Munch(batch_size    = 4,
                       data_path     = None,
                       input_shape   = None,
                       inputs_df     = ['a'],
                       outputs       = ['b'],
                       shuffle       = False,
                       fixed_batches = True)

    # check `fixed_batches` switch
    g = gr.DataGeneratorDisk(ids, **gen_params)
    assert np.array_equal([gen.get_sizes(x) for x in g], 
                          ['([array<4,1>], [array<4,1>])', 
                           '([array<4,1>], [array<4,1>])'])
    assert np.array_equal(g[0][0][0].squeeze(), range(4))

    gen_params.fixed_batches = False
    g = gr.DataGeneratorDisk(ids, **gen_params)
    assert np.array_equal([gen.get_sizes(x) for x in g], 
                          ['([array<4,1>], [array<4,1>])',
                           '([array<4,1>], [array<4,1>])',
                           '([array<2,1>], [array<2,1>])'])
    assert np.array_equal(g[2][0][0].squeeze(), [8, 9])

    # check randomized
    gen_params.shuffle = True
    gen_params.fixed_batches = False # maintain
    g = gr.DataGeneratorDisk(ids, **gen_params)

    # check if it returns all items
    data = list(zip(*list(g)))
    data0 = np.concatenate([l[0] for l in data[0]], axis=0).squeeze()
    data1 = np.concatenate([l[0] for l in data[1]], axis=0).squeeze()
    assert np.array_equal(np.sort(data0), np.arange(10))
    assert np.array_equal(np.sort(data1), np.arange(10))

    # check if randomization is applied, consistently
    num_randoms0 = 0
    num_randoms1 = 0
    for i in range(100):
        g = gr.DataGeneratorDisk(ids, **gen_params)
        data = list(zip(*list(g)))
        data0 = np.concatenate([l[0] for l in data[0]], axis=0).squeeze()
        data1 = np.concatenate([l[0] for l in data[1]], axis=0).squeeze()

        # check consistency
        ids_ = ids.copy()
        ids_.index = ids_.a
        np.array_equal(ids_.loc[data0].b, data1)

        num_randoms0 += not np.array_equal(data0, np.arange(10))
        num_randoms1 += not np.array_equal(data1, np.arange(10))

    # check randomization, at least once
    assert num_randoms0
    assert num_randoms0

    # check deterministic
    gen_params.shuffle       = True
    gen_params.deterministic = np.random.randint(100)
    assert np.array_equal(gr.DataGeneratorDisk(ids, **gen_params)[0], 
                          gr.DataGeneratorDisk(ids, **gen_params)[0])
    
    gen_params.update(fixed_batches = False,
                  shuffle       = True,
                  group_by      = 'c',
                  deterministic = False)

    g = gr.DataGeneratorDisk(ids, **gen_params)
    data = list(zip(*list(g)))
    data = [[l[0] for l in d] for d in data]
    data_conc = [np.concatenate(d, axis=0) for d in data]

    # returns all
    df = pd.DataFrame(np.concatenate(data_conc, axis=1), columns=('a','b'))
    x = df.merge(ids, on='a')
    assert np.all(x.b_x == x.b_y)

    # each batch returns a single group
    ids_ = ids.copy()
    ids_.index = ids_.a
    for i, d in enumerate(data[0]):
        assert ids_.loc[d[0]].c.unique().shape==(1,)
        
def test_accessor_function_numpy_array():
    
    ids = pd.DataFrame(dict(a = range(10), 
                        b = list(np.random.randint(0,10,(10,2,2)))))
    gen_params = Munch(batch_size    = 4,
                       data_path     = None,
                       input_shape   = None,
                       inputs_df     = lambda ids: [ids[['a']].values],
                       outputs       = ['b'],
                       shuffle       = False,
                       fixed_batches = True)

    # test using a function to access data from ids
    # test if data in ids items can be ndarrays
    g = gr.DataGeneratorDisk(ids, **gen_params)
    assert gen.get_sizes(g[0])=='([array<4,1>], [array<4,2,2>])'

    # test if double inputs works
    gen_params.outputs = ['a','a']
    g = gr.DataGeneratorDisk(ids, **gen_params)
    assert gen.get_sizes(g[0])=='([array<4,1>], [array<4,2>])'
    
def test_ids_fn():
    gen_params_local = gen_params.copy()
    ids_local = ids.copy()
    
    def ids_fn():
        ids_local.score = -ids_local.score
        return ids_local

    gen_params_local.ids_fn = ids_fn
    gen_params_local.batch_size = 4
    g = gr.DataGeneratorDisk(ids, **gen_params_local)
    x = g[0][1][0]
    g.on_epoch_end()
    y = g[0][1][0]    
    assert np.array_equal(-x, y)