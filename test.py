import logging as log
import time


def dd():
    log.basicConfig(filename='alexnet.txt', level=log.INFO)
    for i in range(10):
        log.info('sss')
        time.sleep(1)
        print('ss')


if __name__ == '__main__':
    dd()