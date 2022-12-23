 # -*- coding: utf-8 -*-

"""Stuff on threading, multiprocessing and asyncIO
Influenced by superfastpython.com in many ways.
"""

import threading
import multiprocessing
import asyncio


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
    # all coroutines for an event loop run in one thread, and a thread runs in one process
    # by extension, locks are neither thread-safe nor process-safe
    # event loop is a programming construct or design pattern that waits for and dispatches events or messages in a program
    # non-preemptive multitasking - OS never initiates a context switch from a running process to another process. Instead, in order to run multiple applications concurrently, processes voluntarily yield control periodically or when idle or logically blocked.

    # mandatory
    @staticmethod
    async def hello_world():
        print('hello')
        await asyncio.sleep(.01)
        print(asyncio.get_event_loop())
        print(asyncio.get_running_loop())
        print('world')


    # Common deadlocks for asyncio
    #   waits on itself
    #   coroutines wait on each other
    #   fails to release a resource
    #   fail to perform lock ordering.

    @staticmethod
    def deadlock_by_reacquiring_mutex():
        async def task2(lock):
            print('Task2 acquiring lock again...')
            async with lock:
                print('will never get here')
                pass

        async def task1(lock):
            print('Task1 acquiring lock...')
            async with lock:
                await task2(lock)

        async def main():
            lock = asyncio.Lock()
            await task1(lock)

        asyncio.run(main())

    @staticmethod
    def deadlock_by_mutual_await():
        async def task(other):
            print(f'awaiting the task: {other.get_name()}')
            await other

        async def main():
            task1 = asyncio.current_task()
            task2 = asyncio.create_task(task(task1))
            await task(task2)

        asyncio.run(main())

    # demonstration of asyncio streams
    # socket = network endpoint, at runtime


__author__ = 'Petr Panov'
__copyright__ = 'Copyleft 2022, Milky Way'
__credits__ = ['Petr Panov']
__license__ = 'MIT'
__version__ = '1.0.0'
__maintainer__ = 'Petr Panov'
__email__ = 'pvpanov93@gmail.com'
__status__ = "Draft"
