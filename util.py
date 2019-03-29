import numpy as np

def get_image_params(json_path, image_path, key=None, show=True, save_file=False, save_path='local'):
    from  Flo2PlanManager.manager.visual import ManageCOCO
    import os
    import shutil

    manager = ManageCOCO(json_path, image_path)
    ids = manager.getImageIds()
    colors = manager.getColorCategories()
    if key is None:
        idx = [np.random.randint(0, len(ids))]
    elif key == 'ALL':
        idx = ids
        show = False
        save_file = True
    else:
        idx = [manager.searchByKey(key)]


    if save_file:
        if os.path.isdir(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path)

    for id in idx:
        p, ann = manager.getImageAnnotations(id)
        print('Image path:', p)

        return p, ann, colors