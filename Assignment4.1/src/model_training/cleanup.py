import torch
import gc
from skorch.callbacks import Callback 

# cleanup function for resource optimization
#   always clear cache and garbage collect, but silently between epochs
#   after done w/ model, delete it
def cleanup(model, is_final=False):
    # Clear CUDA cache 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
            
    # run garbage collection
    gc.collect()

    # print cleanup and delete model when done
    if is_final:
        print("\n\u2713 -- Cleared Cuda Cache.")
        print("\u2713 -- Collected Garbage.")
        print("\u2713 -- Deleted Model.")
        print("\u2713 -- Cleanup Complete.")
        if model:
            del model
        
# callback to run with the model, will cleanup after each epoch
class EpochCleanupCallback(Callback):
    def on_epoch_end(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        cleanup(model=net, is_final=False) # pass current model for cleanup