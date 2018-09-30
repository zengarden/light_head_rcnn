# encoding: utf-8
"""
@author: si zuo
@contact: 807269961@qq.com
"""

class ADASBasic:
    class_names = ['background', 'car',]
    classes_originID = {'car': 1,}
    num_classes = 2


class ADAS(ADASBasic):
    pass
    # train_root_folder = ''
    # train_source = os.path.join(
    #     config.root_dir, 'data', 'MSADAS/odformat/adas_trainvalmini.odgt')
    # eval_root_folder = ''
    # eval_source = os.path.join(
    #     config.root_dir, 'data', 'MSADAS/odformat/adas_minival2014.odgt')
    # eval_json = os.path.join(
    #     config.root_dir, 'data', 'MSADAS/instances_minival2014.json')


if __name__ == "__main__":
    # adas = ADASIns()
    from IPython import embed
    embed()
