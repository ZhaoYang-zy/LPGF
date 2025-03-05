import logging
import os

import arrow


def get_log(path, exp_name, model, pretrain, test=False):
    root = os.path.dirname(os.path.dirname(__file__))
    if test:
        path = os.path.join(root, path, exp_name, 'log')
        time = arrow.now()
        time = time.format('YYYY-MM-DD-HH-mm-ss')
        name = f'{model}_test_{time}.log'
        path = os.path.join(path, name)

    elif pretrain:
        path = os.path.join(root, path, exp_name, 'log')
        time = arrow.now()
        time = time.format('YYYY-MM-DD-HH-mm-ss')
        name = f'{model}_pretrain_{time}.log'
        path = os.path.join(path, name)

    else:
        path = os.path.join(root, path, exp_name, 'log')
        time = arrow.now()
        time = time.format('YYYY-MM-DD-HH-mm-ss')
        name = f'{model}_train_{time}.log'
        path = os.path.join(path, name)

   
    logging.basicConfig(
        format='%(message)s',
        level=logging.DEBUG,
        filename=path,
        filemode='a')


def printlog(metrics, epoch):
    message = f'epoch:{epoch}'
    for key, value in metrics.items():
        s = f'  {key}:{value}\n'
        message = message + s
    logging.info(message)
