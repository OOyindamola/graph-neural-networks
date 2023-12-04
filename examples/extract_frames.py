from PIL import Image

num_key_frames = 100

root = '/home/oyindamola/Research/graph-neural-networks/examples/experiments/SegregationGNN_NStates12_-021-20220508133345DlAggGNN_K3_clipping__DAGGER_/datasetTrajectories'
with Image.open(root+'/test_global.gif') as im:
    for i in range(num_key_frames):
        im.seek(im.n_frames // num_key_frames * i)
        im.save('{}.png'.format(i))

with Image.open(root+'/test_global.gif') as im:
    for i in range(num_key_frames):
        im.seek(im.n_frames // num_key_frames * i)
        im.save('{}.png'.format(i))
