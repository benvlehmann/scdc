"""This module defines a custom MPI scatter-gather manager. It can be used
independently of the remainder of the code.

"""

import tracemalloc
import gc

import logging
import warnings
import traceback
import sys
try:
    from mpi4py import MPI
except ImportError:
    warnings.warn(
        ("It appears that the mpi4py module is not installed. "
         "You can still import this module for autodoc purposes, "
         "but you will not be able to run with MPI on this platform.")
    )


ROOT_RANK = 0
EXCEPTION_SIGNAL = "exception occurred"
TERMINATE_SIGNAL = "terminate worker"
logging.basicConfig()
LOGGER = logging.getLogger('MPI')
LOGGER.setLevel(logging.DEBUG)


class DistributedTask(object):
    def __init__(self, *args, **kwargs):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.nprocs = self.comm.Get_size()
        if self.rank == ROOT_RANK:
            self.root = True
        else:
            self.root = False

    def run(self, *args, **kwargs):
        if self.root:
            result = self._run_root(*args, **kwargs)
            self._done()
            return result
        else:
            try:
                self._run_worker(*args, **kwargs)
            except BaseException:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                LOGGER.error(
                    "WORKER %d: an unhandled exception occurred" % (
                        self.rank
                    )
                )
                # Print the traceback
                LOGGER.error("WORKER %d: traceback:" % self.rank)
                traceback.print_tb(exc_traceback)
                # Propagate exceptions to root
                self.comm.send(
                    (self.rank, EXCEPTION_SIGNAL), dest=ROOT_RANK, tag=0
                )
                raise

    def _run_root(self, *args, **kwargs):
        raise NotImplementedError

    def _scatter(self, tasks, results):
        LOGGER.error(
            "ROOT: Will distribute %d tasks among %d workers" % (
                len(tasks), self.nprocs - 1
            )
        )
        undispatched_task_ids = list(range(len(tasks)))
        unfinished_task_ids = list(range(len(tasks)))
        # `results` is really a buffer. Pop everything and repopulate
        while len(results) > 0:
            results.pop(0)
        while len(results) < len(tasks):
            results.append(None)
        # Distribute initial tasks
        for worker_rank in range(self.nprocs):
            if worker_rank == ROOT_RANK:
                continue
            try:
                next_task_id = undispatched_task_ids.pop(0)
                next_task = tasks[next_task_id]
            except IndexError:
                break
            self.comm.send(
                (next_task_id, next_task),
                dest=worker_rank,
                tag=0
            )
        # Receive output and redistribute
        tracemalloc.start()
        while len(unfinished_task_ids):
            current, peak = tracemalloc.get_traced_memory()
            LOGGER.error(
                "ROOT: Memory usage: %f MB (peak %f MB)" % (
                    current/1e6, peak/1e6
                )
            )
            LOGGER.error("ROOT: Waiting...")
            worker_rank, message = self.comm.recv(tag=0)
            if message == EXCEPTION_SIGNAL:
                LOGGER.error(
                    "ROOT: received an exception signal from worker %d" % (
                        worker_rank
                    )
                )
                self._done()
                raise RuntimeError
            result_id, result = message
            LOGGER.error(
                "ROOT: Worker %d finished task %d" % (
                    worker_rank, result_id
                )
            )
            unfinished_task_ids.remove(result_id)
            results[result_id] = result
            # Dispatch the next task
            n_rem = len(undispatched_task_ids)
            if n_rem:
                LOGGER.error(
                    "ROOT: There are still %d tasks to distribute" % n_rem
                )
                next_task_id = undispatched_task_ids.pop(0)
                next_task = tasks[next_task_id]
                LOGGER.error(
                    "ROOT: Will give this worker task %d" % next_task_id
                )
                self.comm.send(
                    (next_task_id, next_task),
                    dest=worker_rank,
                    tag=0
                )
        LOGGER.error("All tasks completed")
        tracemalloc.stop()
        return results

    def _run_worker(self, *args, **kwargs):
        self.old_current = 0
        self.old_peak = 0
        #tracemalloc.start()

        def _report_memory(tag):
            """
            current, peak = tracemalloc.get_traced_memory()
            delta_current = current - self.old_current
            delta_peak = peak - self.old_peak
            LOGGER.error(
                (
                    "WORKER %d: Memory usage: %f MB [%s%f MB] |"
                    " peak %f MB [%s%f MB] | %s"
                ) % (
                    self.rank, current/1e6,
                    ('+' if delta_current >= 0 else ''), delta_current/1e6,
                    peak/1e6, ('+' if delta_peak >= 0 else ''), delta_peak/1e6,
                    tag
                )
            )
            self.old_current = current
            self.old_peak = peak
            """
            pass

        while True:
            current, peak = tracemalloc.get_traced_memory()
            _report_memory('loop start')
            LOGGER.error("WORKER %d: waiting for task" % self.rank)
            message = self.comm.recv(source=ROOT_RANK)
            _report_memory('post recv')
            if message == TERMINATE_SIGNAL:
                LOGGER.error("WORKER %d: received term signal" % self.rank)
                break
            else:
                task_id, task = message
            LOGGER.error(
                "WORKER %d: received task %d" % (self.rank, task_id)
            )
            _report_memory('pre func')
            result = self._func(task, *args, **kwargs)
            _report_memory('post func')
            LOGGER.error(
                "WORKER %d: finished task %d, sending results to root" % (
                    self.rank, task_id
                )
            )
            self.comm.send(
                (self.rank, (task_id, result)),
                dest=ROOT_RANK,
                tag=0
            )
            _report_memory('post send')
            # Force garbage collection
            del result
            gc.collect()
            _report_memory('post gc')

        tracemalloc.stop()

    def _func(self, *args, **kwargs):
        """The meat and potatoes"""
        raise NotImplementedError

    def _done(self):
        """Terminate workers"""
        LOGGER.error("ROOT: Broadcasting termination signal")
        for worker_rank in range(1, self.nprocs):
            self.comm.send(TERMINATE_SIGNAL, dest=worker_rank, tag=0)
