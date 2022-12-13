# -*- coding: utf-8 -*-

"""Stuff on threading, multiprocessing and asyncIO
Influenced by superfastpython.com in many ways.
"""

import threading
import multiprocessing


class ThreadingTools:
    # https://superfastpython.com/threading-in-python/
    # Thread(target=.., args=.., kwargs=..)
    # .start() and .join() (explicit wait) methods.
    # inherit (extend) class to create concurrent threads
    # .current_thread() gives object with attributes name, daemon, ident
    class CustomThread(threading.Thread):
        def __init__(self, value):
            threading.Thread.__init__(self)
            self.value = value
            
        # override

        def run(self):
            thr = threading.current_thread()
            print(f'This is coming from another thread: {self.value} {thr.name} {thr.daemon} {thr.ident}')


class MultiProcessingTools:
    # https://superfastpython.com/multiprocessing-in-python/
    class CustomProcess(multiprocessing.Process):
        def __init__(self, value):
            multiprocessing.Process.__init__(self)
            self.value = value
            
        # override

        def run(self):
            proc = multiprocessing.current_process()
            print(f'This is coming from another thread: {self.value} {proc.name} {proc.daemon} {proc.ident}')


class AsyncIOTools:
    # https://superfastpython.com/python-asyncio/
    pass


__author__ = 'Petr Panov'
__copyright__ = 'Copyleft 2022, Milky Way'
__credits__ = '[Petr Panov]'
__license__ = 'MIT'
__version__ = '1.0.0'
__maintainer__ = 'Petr Panov'
__email__ = 'pvpanov93@gmail.com'
__status__ = "Draft"
