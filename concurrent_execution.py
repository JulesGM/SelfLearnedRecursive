
"""

Table of Contents:
===================
    > `actual_prediction`:
        - Prediction function called by the worker thread and by the async stuff.
    > `make_thread_safe_pipes`:
        - Contructs the two Comm objects.
    > `BasicPredFunctor`:
        - Functor called by the nodes to run generation directly.
    > `SendPull`:
        - Pipe interface specific to our application.
    > `SendPullPredFunctor`:
        - Function called by the nodes to send the predictions to the worker node.
    > `pool_work`:
        - Can be seen as the clients of the server-client system, with the worker being the server.
    > `multi_worker`:
        - This is the function for the worker thread.
    > `start`:
        - Main function, starts the multithreaded approach.

Thoughts:
=========
    > `actual_prediction` should be elsewhere
    > Basically all the names should be reworked.
    > Maybe writing down why we think it's not working here would be better.

"""
import dataclasses
import multiprocessing
import multiprocessing.dummy as dummy
import queue
import random
import threading
import time

import numpy as np
import rich
import torch
from tqdm import tqdm

import datagen

def actual_prediction(batch, collator, model, generation_kwargs, gen_function, max_answer_gen):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Use the Data Collator on the data
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    batch = collator(batch)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Cudify everything
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for k, v in batch.items():
        batch[k] = v.cuda()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create the decoder_attention_mask
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if "decoder_attention_mask" in batch:
        batch["decoder_attention_mask"] = batch["decoder_attention_mask_for_gen"]
        del batch["decoder_attention_mask_for_gen"]

    bound_length = min(model.config.max_length - 6, batch["decoder_input_ids_for_gen"].shape[1] - 1)
    batch["decoder_input_ids"] = batch["decoder_input_ids_for_gen"][:, :bound_length] # TODO: LENGTH STUFF
    del batch["decoder_input_ids_for_gen"]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    start = time.perf_counter()
    output = gen_function(
        model=model,
        **batch,
        **generation_kwargs,
        max_length=max_answer_gen + batch["input_ids"].shape[1]
    )

    delta = time.perf_counter() - start
    
    # print(f"Generation took {delta:.5f} seconds, {delta / output.shape[0]}s per item.")
    return output

def make_thread_safe_pipes():
    a = queue.Queue()
    b = queue.Queue()
    
    @dataclasses.dataclass
    class Comm:
        send_q: queue.Queue
        recv_q: queue.Queue
        def send(self, x):
            self.send_q.put(x)
        
        def recv(self):
            return self.recv_q.get()

        def poll(self):
            return self.recv_q.qsize() > 0

    return Comm(a, b), Comm(b, a)


class BasicPredFunctor:
    __slots__ = (
        "tokenizer", 
        "model"
    )

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def __call__(self, input_str: str, pseudo_without_head: str):
        DEVICE = "cuda"
        input_ids = self.tokenizer(input_str, return_tensors="pt")
        decoder_input_ids = self.tokenizer(pseudo_without_head, return_tensors="pt")
        output = self.model.to(DEVICE).generate(
            input_ids=input_ids.to(DEVICE).reshape(1, -1), 
            decoder_input_ids=decoder_input_ids.to(DEVICE).reshape(1, -1),
        )
        assert len(output) == 1, output.shape
        list_output = output[0].cpu().detach().numpy().tolist()
        list_output_filtered = list_output[len(input_ids):]
        good_output = self.tokenizer.decode(list_output_filtered)
        if good_output[-1] == ")":
            good_output = good_output[:-1]
        good_output = good_output.replace("<eos>", "")
        return good_output.strip()

class SendPull:

    __slots__ = (
        "_sender",
        "_puller",
    )

    def __init__(self, pipe_type):
        self._sender, self._puller = pipe_type()
    
    def sender_send_recv(self, item, start_sending_message, sent_message, received_message):
        rich.print(start_sending_message)
        self._sender.send(item)
        rich.print(sent_message)
        received_package = self._sender.recv()
        rich.print(received_message)
        return received_package

    def puller_is_empty(self):
        return not self._puller.poll()

    def puller_pull(self):
        package_received = self._puller.recv()
        return package_received

    def puller_send(self, item):
        self._puller.send(item)


class SendPullPredFunctor:
    """
    This thing is passed to the prediction based dataset. 
    It is used to send the data to a worker thread, that packs them in a batch, and
    does inference on multiple samples at once.
    """
    __slots__ = (
        "_tokenizer",
        "_send_pull"
    )

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._send_pull = None

    def set_send_pull(self, send_pull):
        self._send_pull = send_pull

    def __call__(self, input_str: str, pseudo_without_head: str, pred_logger: datagen.PredLogger):
        """
        Table of contents:
        1. Prepare the data for the generation thread
        2. Send the data to the generation thread and wait for the results
        3. Make sure we got the correct package back
        4. Prepare the data to be returned
        """
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 1. Prepare the data for the generation thread
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        header = threading.current_thread().name
        input_ids, decoder_input_ids = datagen.prep_input_data(input_str, pseudo_without_head)
        verification_code = threading.current_thread().name + str(random.random())

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2. Send the data to the generation thread, wait for the results
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        do_str = pred_logger.log_doing(input_str, pseudo_without_head)
        start_sending_str =     f"[bright_cyan]({header}) - CLIENT STARTED_SENDING:     {do_str}"
        sent_str =              f"[bright_cyan]({header}) - CLIENT SENT:                {do_str}"
        received_str =          f"[bright_cyan]({header}) - CLIENT RECEIVED:            {do_str}"
        received = self._send_pull.sender_send_recv(
            dict(
                payload=dict(
                    input_ids=input_ids, 
                    decoder_input_ids=decoder_input_ids,
                ),
                verification_code=verification_code
            ),
            start_sending_message=start_sending_str,
            sent_message=sent_str, 
            received_message=received_str
        )

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 3. Make sure we got the correct package back
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        input_ids_verif = received["payload"]["input_ids"]
        assert received["verification_code"] == verification_code, (
            (received["verification_code"], verification_code), 
            (input_ids, input_ids_verif),
        )
        assert np.all(input_ids_verif == input_ids), (
            input_ids_verif, input_ids
        )
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 4. Prepare the data to be returned
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        output = received["payload"]["output"]
        return datagen.prep_input_data.prep_return_data(output, decoder_input_ids, self._tokenizer)

def pool_work(things):
    """
    Used in `_make_dataloader`, as the function used 
    """
    idx, dl_name, send_pulls, ds, num_workers = things
    # worker_id = int(threading.current_thread().name.split("_")[-1])
    # worker_id = int(dummy.current_process().name.split("-")[-1])
    worker_id = idx % num_workers
    rich.print("[green bold]", f"({dl_name} - {worker_id}) - Starting. SENDPULL")
    ds.set_send_pull(send_pulls[worker_id])
    return ds.get(idx)    

def multi_worker(dl_name, send_pulls, done, loop_wait_sec, collator, model, tokenizer, generation_kwargs):
    thread_namme = threading.current_thread().name
    header = f"({dl_name} - {thread_namme}) - "
    this_it = []
    while not done[0]:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Stack eventual new send-pull-requests onto the work pile
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for send_pull_idx, send_pull in enumerate(send_pulls):
            if not send_pull.puller_is_empty():
                received = send_pull.puller_pull()
                payload = received["payload"]
                verification_code = received["verification_code"]
                this_it.append((send_pull_idx, payload, verification_code))
        
        if len(this_it) >= 1: # Ideal would be to know how many nodes are left
            rich.print(f"[green bold]{header}Unpacking. [red]{len(this_it) = }[green], {len(send_pulls) = }")
            batch = [x for _, x, _ in this_it]
            output = actual_prediction(batch, collator, model, tokenizer, generation_kwargs)

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Send the results back
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            for idx_in_batch, (send_pull_idx, payload, verification_code) in enumerate(this_it):
                assert np.all(batch["input_ids"][idx_in_batch].detach().cpu().numpy() == payload["input_ids"]), (
                    batch["input_ids"][idx_in_batch], payload["input_ids"]
                )
                
                send_pulls[send_pull_idx].puller_send(
                    dict(
                        payload=dict(
                            output=output[idx_in_batch].detach().cpu().numpy(), 
                            input_ids=batch["input_ids"][idx_in_batch].detach().cpu().numpy(), # For verification purposes
                        ),
                        verification_code=verification_code
                    )
                )

            this_it = []
        else:
            # rich.print(f"[blue]{header}sleeping - {len(this_it)}")
            time.sleep(loop_wait_sec)

    rich.print("[green bold]WORKER DONE")

def start(
    indices, 
    batch_size, 
    pipe_type, 
    dl_name, 
    ds, 
    collator, 
    model, 
    tokenizer, 
    generation_kwargs, 
    loop_wait_sec, 
    pool_constructor,
):
    num_workers = batch_size
    send_pulls = [SendPull(pipe_type) for _ in range(num_workers)]
    done = [False]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build and start the worker thread that pulls from the queues
    # and does inference
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    threading.Thread(
        target=lambda: 
            multi_worker(
                dl_name=dl_name, 
                send_pulls=send_pulls, 
                done=done, 
                loop_wait_sec=loop_wait_sec, 
                collator=collator, 
                model=model, 
                tokenizer=tokenizer, 
                generation_kwargs=generation_kwargs
            )
        ).start()
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Make worker pool and map over batch ids, getting from ds
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    with pool_constructor(num_workers) as pool:
        for i in range(0, len(ds), batch_size):
            mapped = tqdm(pool.map(
                pool_work, 
                [(idx, dl_name, send_pulls, ds, num_workers)  
                for idx in indices[i:i + batch_size]]
            ), total=len(indices) // batch_size)
            listed = list(mapped)
            collated = collator(listed)
            yield collated